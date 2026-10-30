"""Turn what a patient types, or a symptom name or answer from the Agnos data, into KB ids.

A symptom text goes through these layers, cheapest first (layer numbers as in the README):

    [1] clean       lower case, no spaces, no ๆ, "ปวดดดด" -> "ปวด";   "ไม่มีไข้" -> fever, absent
    [3] exact       a KB Thai name, alias or English name              ตัวร้อน -> fever
        learned     a term learned from the training data (learn.py)   หายใจทางจมูกไม่ได้ -> nasal_congestion
    [3] fillers     the same after dropping filler words               มีเสมหะ -> sputum
    [4] body part   body part x sensation, in either order             ท้องแสบ -> burning_abdominal_pain
    [2] split       several symptoms typed together                    อาเจียนแสบท้อง -> vomiting
                                                                       + burning_abdominal_pain
    [5] fuzzy       typos: close spelling, or the same Thai sound      ปวดหอง -> abdominal_pain
    [6] semantic    optional local retrieval after an unmatched result; train-calibrated acceptance

If nothing is close enough, the text is returned unmatched with the closest symptoms as candidates,
so the app can ask the patient to choose instead of guessing. So are phrases the lexicon marks as
ambiguous ("หายใจไม่สะดวก": blocked nose or shortness of breath).

An HPI answer ("ระยะเวลา 2 สัปดาห์") is read for one symptom with read_answer().
"""
import re
from difflib import SequenceMatcher

from pythainlp.corpus.common import thai_words
from pythainlp.soundex import soundex
from pythainlp.tokenize import word_tokenize
from pythainlp.util import Trie

from ..kb.schema import History
from .assertions import assertion_clauses
from .text import QUESTION_UNIT_DAYS, categories_for, has_thai, key, read_duration, read_temperature

# Thresholds, chosen on the training set.
FUZZY_MIN = 0.85        # spelling similarity (0-1) that is accepted on its own
SAME_SOUND_MIN = 0.7    # lower similarity accepted when the Thai sound code is also the same
SAME_SOUND_MAX_LENGTH = 8   # the sound code covers only the first few sounds, so it is used for
                            # short texts only ("ปวดหอง"), not long ones that share a start
FUZZY_MARGIN = 0.05     # the best name must beat the closest other symptom by this much
LEARNED_SCORE = 0.95
BODY_PART_SCORE = 0.9
SPLIT_SCORE = 0.9       # every word was part of a symptom
SPLIT_PARTIAL_SCORE = 0.7   # some words were left over ("ไอตอนตีสอง": ตอนตีสอง)
JOINING_WORDS = ["และ", "กับ", "หรือ", "ร่วมกับ", "and", "or", "with"]


class Match:
    def __init__(self, text, symptom, layer, score, absent=False, candidates=None, state=None):
        self.text = text                      # the words it came from
        self.symptom = symptom                # KB symptom id, or None if nothing was close enough
        self.layer = layer      # rule layer, semantic, semantic_candidates, ambiguous, or none
        self.score = score                    # 1.0 for exact, lower for the looser layers
        self.state = state or ("absent" if absent else "present")
        # Other symptoms the words may mean. With no symptom: the ones to choose from, best first.
        # With a symptom: alternatives worth asking about ("แสบคอ": heartburn, sore throat too).
        self.candidates = candidates or []

    def __repr__(self):
        return f"Match({self.text!r} -> {self.symptom}, {self.layer}, {self.score:.2f}, state={self.state})"

    @property
    def absent(self):
        """Compatibility for callers specifically checking a definite negative."""
        return self.state == "absent"


class Normalizer:
    def __init__(self, kb, lexicon, learned=None, semantic=None):
        self.kb = kb
        self.lexicon = lexicon
        self.learned = learned or {}    # key of a term learned from training data -> symptom id
        self.semantic = semantic

        self.fillers_start = longest_first(lexicon.fillers_start)
        self.fillers_end = longest_first(lexicon.fillers_end)
        self.negations = longest_first(lexicon.negations)

        # key of every Thai name, alias and English name -> symptom id
        self.names = {}
        for symptom in kb.symptoms.values():
            for name in [symptom.th, symptom.en] + symptom.aliases:
                self.names.setdefault(key(name), symptom.id)
        # a history item is also known without its leading word: "เป็นเบาหวาน" -> "เบาหวาน",
        # so "ประวัติเบาหวาน" (filler "ประวัติ" dropped) finds it
        for symptom in kb.symptoms.values():
            if symptom.kind == "history":
                for name in [symptom.th] + symptom.aliases:
                    short = self.strip_fillers(key(name))
                    if short and short not in self.names:
                        self.names[short] = symptom.id

        # Thai sound code of each Thai name, for typos
        self.sounds = {}
        for name in self.names:
            if has_thai(name):
                self.sounds[name] = soundex(name, engine="lk82")

        # body part word -> BodyPart;  sensation words, longest first, as (word, sensation)
        self.body_part_words = {}
        for part in lexicon.body_parts.values():
            for word in part.words:
                self.body_part_words[key(word)] = part
        self.sensation_words = []
        for sensation, words in lexicon.sensations.items():
            for word in words:
                self.sensation_words.append((key(word), sensation))
        self.sensation_words.sort(key=word_length, reverse=True)

        # key of an ambiguous phrase -> the symptoms it can mean
        self.ambiguous = {}
        for phrase, symptom_ids in lexicon.ambiguous.items():
            self.ambiguous[key(phrase)] = symptom_ids

        # Word lists for cutting Thai text into words (pythainlp, "newmm"):
        #   thai_and_ours  a general Thai dictionary + symptom names + lexicon words: the first cut
        #   ours           symptom names + lexicon words only: cuts one word further ("ผื่นคัน")
        #   thai_only      a general Thai dictionary + lexicon words, no symptom names: compares the
        #                  words inside names ("ปวดรอบสะดือ" -> ปวด + รอบ + สะดือ)
        lexicon_words = set(self.body_part_words) | set(self.negations) | set(self.fillers_start)
        for word, _ in self.sensation_words:
            lexicon_words.add(word)
        general = set(thai_words())
        self.thai_and_ours = Trie(general | lexicon_words | set(self.names) | set(self.learned))
        self.ours = Trie(lexicon_words | set(self.names) | set(self.learned))
        self.thai_only = Trie(general | lexicon_words)

    # ── symptoms ───────────────────────────────────────────────────────────
    def read_symptoms(self, text):
        """All symptoms in a text, as a list of Match. Commas separate symptoms; so do "/" and
        brackets, unless the whole piece is a name ("ear discharge/drainage" -> "ear discharge")."""
        matches = []
        for piece in text.replace("\n", ",").split(","):
            if not key(piece):
                continue
            if re.search(r"[/()]", piece) and self.lookup(key(piece)) is None:
                for part in re.split(r"[/()]", piece):
                    if key(part):
                        matches += self.read_assertions(part)
            else:
                matches += self.read_assertions(piece)
        return matches

    def read_assertions(self, text):
        # Preserve complete names such as "ไม่มีแรง" and established history labels.
        if key(text) in self.names or key(text) in self.ambiguous:
            return self.read_piece(text)
        matches = []
        for clause, state in assertion_clauses(text):
            read = self.read_piece(clause)
            if state:
                for match in read:
                    match.state = state
            matches.extend(read)
        return matches

    def symptom_ids(self, text):
        """Just the ids of the symptoms the text says the patient has."""
        ids = []
        for match in self.read_symptoms(text):
            if match.symptom and match.state == "present" and match.symptom not in ids:
                ids.append(match.symptom)
        return ids

    def read_piece(self, piece):
        text = key(piece)
        absent = False
        # A complete KB name may contain a lexical negative (e.g. inability to do something).
        if text in self.names:
            return [Match(piece, self.names[text], "exact", 1.0)]
        for word in self.negations:
            if text.startswith(word) and len(text) > len(word):
                text = text[len(word):]
                absent = True
                break

        if text in self.ambiguous:
            return [Match(piece, None, "ambiguous", 0.0, absent, self.ambiguous[text])]

        found = self.lookup(text)
        if found:
            symptom, layer, score, alternatives = found
            return [Match(piece, symptom, layer, score, absent, alternatives)]

        if has_thai(text):
            words = self.tokens(text, self.thai_and_ours)
        else:
            words = key_words(piece)
        matches = self.split(text, words, absent)
        if matches:
            return self.semantic_fallback(piece, text, absent, matches)

        return self.semantic_fallback(piece, text, absent,
                                      [self.fuzzy(piece, self.strip_fillers(text), absent)])

    def semantic_fallback(self, piece, text, absent, matches):
        # Preserve every existing assignment, negation and explicit ambiguity decision.
        if self.semantic is None or any(m.symptom for m in matches):
            return matches
        ranked = self.semantic.search(text if absent else piece)
        if not ranked:
            return matches
        candidates = [symptom_id for symptom_id, _ in ranked]
        if self.semantic.accepts(ranked):
            symptom_id, score = ranked[0]
            return [Match(piece, symptom_id, "semantic", score, absent, candidates[1:])]
        return [Match(piece, None, "semantic_candidates", ranked[0][1], absent, candidates)]

    def lookup(self, text):
        """By name, learned term, name without fillers, or body part:
        (symptom id, layer, score, alternatives), or None."""
        if text in self.names:
            return self.names[text], "exact", 1.0, []
        if text in self.learned:
            return self.learned[text], "learned", LEARNED_SCORE, []
        stripped = self.strip_fillers(text)
        if stripped and stripped in self.names:
            return self.names[stripped], "fillers", 1.0, []
        if stripped and stripped in self.learned:
            return self.learned[stripped], "learned", LEARNED_SCORE, []
        symptom_ids = self.body_part(stripped)
        if symptom_ids:
            return symptom_ids[0], "body_part", BODY_PART_SCORE, symptom_ids[1:]
        return None

    def strip_fillers(self, text):
        for word in self.fillers_start:
            if text.startswith(word):
                text = text[len(word):]
                break
        for word in self.fillers_end:
            if text.endswith(word):
                text = text[: len(text) - len(word)]
                break
        return text

    def body_part(self, text):
        """Body part x sensation in either order: "ท้องแสบ", "แสบท้อง" -> [burning_abdominal_pain].

        Returns the symptom ids the lexicon gives (the first is the reading, the rest alternatives),
        or None.
        """
        sensations = []
        changed = True
        while changed:
            changed = False
            for word, sensation in self.sensation_words:
                if text.startswith(word) and len(text) > len(word):
                    text = text[len(word):]
                elif text.endswith(word) and len(text) > len(word):
                    text = text[: len(text) - len(word)]
                else:
                    continue
                sensations.append(sensation)
                changed = True
                break
        part = self.body_part_words.get(text)
        if part is None or not sensations:
            return None

        # "ปวดแสบท้อง": a more specific sensation (burning) wins over pain;
        # "เจ็บ" (sore) reads like "ปวด" (pain) unless the body part says otherwise
        reading = None
        for sensation in sensations:
            if sensation not in ("pain", "sore") and getattr(part, sensation):
                reading = getattr(part, sensation)
                break
        if reading is None and "sore" in sensations and part.sore:
            reading = part.sore
        if reading is None and ("pain" in sensations or "sore" in sensations):
            reading = part.pain
        if reading is None:
            return None
        if isinstance(reading, str):
            return [reading]
        return reading

    def split(self, text, words, absent):
        """Several symptoms typed together: runs of words that are symptom names.

        - A word that is not a symptom is cut further only if every piece is a symptom:
          "ผื่นคัน" is ผื่น + คัน, but "ไอศกรีม" (ice cream) is not ไอ + ศกรีม.
        - Words left over lower the score: "ไอตอนตีสอง" -> cough, "ตอนตีสอง" left over.
        - A body part left over means the text names a place the lexicon cannot pair with the
          symptom found ("คันที่หลัง" is not just "คัน"), so the symptoms found are only offered as
          candidates.
        Returns [] when no symptom is found.
        """
        start_absent = absent
        matches = []
        leftover = []
        i = 0
        while i < len(words):
            found, end = self.longest_name(words, i)
            if found:
                matches.append(Match("".join(words[i:end]), found[0], "split", 0.0, absent, found[3]))
                absent = False
                i = end
                continue
            pieces = self.all_pieces_known(words[i])
            if pieces:
                for symptom_id in pieces:
                    matches.append(Match(words[i], symptom_id, "split", 0.0, absent))
                absent = False
            elif words[i] in self.negations:
                absent = True
            elif words[i] not in JOINING_WORDS and words[i] not in self.fillers_start:
                leftover.append(words[i])
            i += 1
        if not matches:
            return []

        matches = drop_parents(matches, self.kb)
        found_ids = []
        for m in matches:
            if m.symptom not in found_ids:   # "กลืนลำบาก กลืนติด": two names of one symptom
                found_ids.append(m.symptom)
        for word in leftover:
            if word in self.body_part_words:
                return [Match(text, None, "none", 0.0, start_absent, found_ids)]

        score = SPLIT_SCORE
        if leftover:
            score = SPLIT_PARTIAL_SCORE
        kept = []
        for m in matches:
            if m.symptom in found_ids:
                found_ids.remove(m.symptom)
                m.score = score
                kept.append(m)
        if len("".join(leftover)) >= 2:
            kept.append(Match("".join(leftover), None, "none", 0.0))
        return kept

    def longest_name(self, words, start):
        """The longest run of words from `start` that lookup() knows: (found, end) or (None, start)."""
        end = len(words)
        while end > start:
            found = self.lookup("".join(words[start:end]))
            if found:
                return found, end
            end -= 1
        return None, start

    def all_pieces_known(self, word):
        """Symptom ids if the word cuts into pieces that are all symptoms, else []."""
        if not has_thai(word):
            return []
        pieces = self.tokens(word, self.ours)
        ids = []
        i = 0
        while i < len(pieces):
            found, end = self.longest_name(pieces, i)
            if not found:
                return []
            ids.append(found[0])
            i = end
        return ids

    def tokens(self, text, dictionary):
        """Thai text cut into words with the given word list."""
        return word_tokenize(text, custom_dict=dictionary, engine="newmm", keep_whitespace=False)

    def fuzzy(self, piece, text, absent):
        """The closest name by spelling, helped by Thai sound for typos like ปวดหอง.

        Accepted only if it is clearly closer than the closest name of another symptom;
        otherwise the closest symptoms are offered as candidates.
        """
        sound = None
        if has_thai(text) and len(text) <= SAME_SOUND_MAX_LENGTH:
            sound = soundex(text, engine="lk82")
        scored = []
        for name, symptom in self.names.items():
            similarity = SequenceMatcher(None, text, name).ratio()
            same_sound = sound is not None and self.sounds.get(name) == sound
            scored.append((same_sound and similarity >= SAME_SOUND_MIN, similarity, symptom))
        scored.sort(reverse=True)

        best_sound, best_similarity, best_symptom = scored[0]
        accepted = best_similarity >= FUZZY_MIN or best_sound
        for other_sound, other_similarity, other_symptom in scored[1:]:
            if other_symptom != best_symptom:
                if other_sound == best_sound and best_similarity - other_similarity < FUZZY_MARGIN:
                    accepted = False
                break
        if accepted:
            return Match(piece, best_symptom, "fuzzy", best_similarity, absent)

        candidates = []
        for _, _, symptom in scored:
            if symptom not in candidates:
                candidates.append(symptom)
            if len(candidates) == 3:
                break
        return Match(piece, None, "none", 0.0, absent, candidates)

    # ── HPI answers ────────────────────────────────────────────────────────
    def read_answer(self, symptom_id, text):
        """Read current HPI only; uncertain/past clauses stay unresolved for the caller."""
        result = {}
        for clause, state in assertion_clauses(text):
            if state is not None:
                continue
            for question, values in self.read_current_answer(symptom_id, clause).items():
                for value in values:
                    add_answer(result, question, value)
        return result

    def read_current_answer(self, symptom_id, text):
        """The HPI answers in one answer text, as {"<symptom>.<question>": [answers]}.

            ("abdominal_pain", "ปวดแถวๆรอบสะดือ") -> {"abdominal_pain.location": ["periumbilical"]}
            ("cough", "ไอมา 2 สัปดาห์")          -> {"cough.duration": ["acute"]}
        """
        root = self.root(symptom_id)
        result = {}

        # A more specific form of the symptom named in the answer, e.g. "รอบสะดือ" for ปวดท้อง
        for child in self.children_named(root, text):
            for question, answer in self.kb.symptoms[child].refines.items():
                add_answer(result, root + "." + question, answer)

        # Numbers: how long, or how hot
        for name, question in self.kb.questions.items():
            if not name.startswith(root + "."):
                continue
            if question.unit in QUESTION_UNIT_DAYS:
                span = read_duration(text, self.lexicon)
                if span:
                    per_unit = QUESTION_UNIT_DAYS[question.unit]
                    for answer in categories_for(question, span[0] / per_unit, span[1] / per_unit):
                        add_answer(result, name, answer)
            elif question.unit == "celsius":
                span = read_temperature(text)
                if span:
                    for answer in categories_for(question, span[0], span[1]):
                        add_answer(result, name, answer)
        return result

    def children_named(self, symptom_id, text):
        """Children of a symptom whose own words all appear in the text.

        A child's own words are its name minus the parent's words and sensation words:
        periumbilical_pain "ปวดรอบสะดือ" minus "ปวดท้อง" -> {"รอบ", "สะดือ"}.
        Each own word is looked for anywhere in the text, because Thai word cutting can
        cut the same words differently inside a longer sentence.
        """
        text_key = key(text)
        text_words = set(self.words(text))
        parent = self.kb.symptoms[symptom_id]
        parent_words = set()
        for name in [parent.th, parent.en] + parent.aliases:
            parent_words |= set(self.words(name))
        for word, _ in self.sensation_words:
            parent_words.add(word)

        found = []
        for child_id in self.kb.children.get(symptom_id, []):
            child = self.kb.symptoms[child_id]
            if not child.refines:
                continue
            for name in [child.th, child.en] + child.aliases:
                own = set(self.words(name)) - parent_words
                if own and all_appear(own, text_key, text_words, self.negations):
                    found.append(child_id)
                    break
        return found

    def words(self, text):
        """Words for comparing names: Thai is cut with the dictionary, English at spaces."""
        if has_thai(text):
            return self.tokens(key(text), self.thai_only)
        return key_words(text)

    def root(self, symptom_id):
        while self.kb.symptoms[symptom_id].parent:
            symptom_id = self.kb.symptoms[symptom_id].parent
        return symptom_id

    # ── whole sessions ─────────────────────────────────────────────────────
    def read_session(self, session):
        """A data session in KB ids: (chief complaint id or None, History)."""
        return self.read_history(session.confirmed(), session.answers, session.absent, session.unsure)

    def read_history(self, names, answers, absent_names, unsure_names=()):
        """Symptom names (the first is the chief complaint), their answers and the names the
        patient said no to, in KB ids: (chief complaint id or None, History).

        Names that cannot be placed are kept in history.unmapped.
        """
        chief_complaint = None
        states = {"present": [], "absent": [], "unsure": [], "historical": []}
        unmapped = []
        hpi = {}
        for default, terms in [("present", names), ("absent", absent_names), ("unsure", unsure_names)]:
            for n, name in enumerate(terms):
                matches = self.read_symptoms(name)
                ids = []
                for match in matches:
                    state = match.state if match.state != "present" else default
                    if match.symptom:
                        if match.symptom not in states[state]:
                            states[state].append(match.symptom)
                        if state == "present":
                            ids.append(match.symptom)
                if (default == "present" and not any(m.symptom for m in matches)
                        and all(m.state == "present" for m in matches)):
                    unmapped.append(name)
                if default == "present" and n == 0 and ids:
                    chief_complaint = ids[0]
                if ids:
                    for answer in answers.get(name, []):
                        for question, read in self.read_answer(ids[0], answer).items():
                            for one in read:
                                add_answer(hpi, question, one)
        history = History(hpi=hpi, **states, unmapped=unmapped)
        return chief_complaint, history


def drop_parents(matches, kb):
    """If both a symptom and a more specific form of it were found, keep only the specific one."""
    found = set()
    for m in matches:
        found.add(m.symptom)
    kept = []
    for m in matches:
        more_specific = False
        for other in found:
            if other != m.symptom and is_ancestor(m.symptom, other, kb):
                more_specific = True
        if not more_specific:
            kept.append(m)
    return kept


def is_ancestor(ancestor, symptom_id, kb):
    parent = kb.symptoms[symptom_id].parent
    while parent:
        if parent == ancestor:
            return True
        parent = kb.symptoms[parent].parent
    return False


def all_appear(words, text_key, text_words, negations):
    """Thai words anywhere in the text, but not right after a negation ("ไม่หมุน" is not "หมุน");
    English words as whole words ("ear" is not in "near")."""
    for word in words:
        if has_thai(word):
            if not appears_not_negated(word, text_key, negations):
                return False
        elif word not in text_words:
            return False
    return True


def appears_not_negated(word, text, negations):
    start = text.find(word)
    while start != -1:
        negated = False
        for negation in negations:
            if text[:start].endswith(negation):
                negated = True
        if not negated:
            return True
        start = text.find(word, start + 1)
    return False


def add_answer(result, question, answer):
    answers = result.setdefault(question, [])
    if answer not in answers:
        answers.append(answer)


def longest_first(words):
    keys = []
    for word in words:
        keys.append(key(word))
    keys.sort(key=len, reverse=True)
    return keys


def word_length(pair):
    return len(pair[0])


def key_words(text):
    """English words of a text, each in key form: "Dry cough" -> ["dry", "cough"]."""
    words = []
    for word in text.lower().split():
        if key(word):
            words.append(key(word))
    return words
