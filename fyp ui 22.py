import gradio as gr
import json
from unicodedata import normalize
import queue
from vosk import Model, KaldiRecognizer
import sounddevice as sd
from fuzzywuzzy import fuzz
import time
from collections import defaultdict

# Load Quran data
def load_quran(file_path):
    quran = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) == 3:
                    surah, ayah, text = parts
                    surah = int(surah)
                    ayah = int(ayah)
                    if surah not in quran:
                        quran[surah] = {}
                    quran[surah][ayah] = text
    except Exception as e:
        print(f"Error loading Quran file: {e}")
    return quran

# Load Surah names from file
def load_surah_names(file_path):
    surah_names = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(':')
                if len(parts) == 2:
                    surah_num = int(parts[0].strip())
                    surah_name = parts[1].strip()
                    surah_names[surah_num] = {
                        "en": f"Surah {surah_num}",
                        "ar": surah_name
                    }
    except Exception as e:
        print(f"Error loading surah names file: {e}")
        # Fallback to default names if file can't be loaded
        for i in range(1, 115):
            surah_names[i] = {"en": f"Surah {i}", "ar": f"سورة {i}"}
    return surah_names

quran = load_quran("E:/FYP/quran-simple.txt")
surah_names = load_surah_names("E:/FYP/surah_mapping_arabic.txt")
model = Model("E:/FYP/vosk-model-ar-0.22-linto-1.1.0")
rec = KaldiRecognizer(model, 16000)
rec.SetWords(True)

# Global State
q = queue.Queue()
state = {
    "surah": 1,
    "ayah": 1,
    "buffer": "",
    "running": False,
    "recited_ayahs": {},
    "stop_requested": False,
    "current_attempt": {},
    "partial_result": "",
    "last_update": "",
    "surah_content": "",
    "errors": defaultdict(list),  # Track errors per ayah
    "recited_text": defaultdict(str),  # Track what was actually recited
    "expected_text": defaultdict(str)  # Track what was expected
}

accuracy_threshold = 67 # Increased threshold for better accuracy

def get_ayah(surah, ayah):
    return quran.get(surah, {}).get(ayah, "")

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(bytes(indata))

def normalize_arabic(text):
    """Normalize Arabic text to standard form and group similar diacritics"""
    text = normalize('NFC', text)  # Normalize to composed form
    
    # Define groups of similar diacritics that should be considered equivalent
    similar_diacritics = {
        'َ': ['ً'],  # Fatha and Fathatan are similar
        'ِ': ['ٍ'],  # Kasra and Kasratan are similar
        'ُ': ['ٌ'],  # Damma and Dammatan are similar
        'ْ': [],     # Sukun
        'ّ': [],    # Shadda
        'َ': 'ا',  # Fatha (Zabr) → Alif
        'ً': 'ا',  # Fathatan → Alif
        'ا': 'ا',  # Alif remains Alif
    }
    
    # Replace similar diacritics with their base form
    replacements = {}
    for base, equivalents in similar_diacritics.items():
        for equiv in equivalents:
            replacements[equiv] = base
    
    # Apply the replacements
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove tatweel (elongation character)
    text = text.replace('ـ', '')
    
    return text

def calculate_similarity(expected, recited):
    """Improved similarity calculation for Arabic with diacritics"""
    # Normalize both texts (this will now group similar diacritics)
    expected_norm = normalize_arabic(expected)
    recited_norm = normalize_arabic(recited)
    
    # If they match exactly after normalization
    if expected_norm == recited_norm:
        return 100
    
    # Calculate base similarity (without any diacritics)
    expected_base = ''.join([c for c in expected_norm if not (0x64B <= ord(c) <= 0x652)])
    recited_base = ''.join([c for c in recited_norm if not (0x64B <= ord(c) <= 0x652)])
    
    # If base letters don't match, return regular similarity
    if expected_base != recited_base:
        return fuzz.ratio(expected_norm, recited_norm)
    
    # If base letters match, be more lenient with diacritics
    base_similarity = 80  # High base score since letters match
    diacritic_similarity = fuzz.ratio(expected_norm, recited_norm)
    
    # Weighted average favoring base letters
    return int(base_similarity * 0.7 + diacritic_similarity * 0.3)
def highlight_words(expected, recited, accuracy_threshold=accuracy_threshold, current_word_index=None):
    expected_words = expected.split()
    recited_words = recited.split()

    highlighted = []
    accuracy_count = 0
    error_details = []
    
    for i, e in enumerate(expected_words):
        # Default to uncolored text
        word_style = ""
        
        # Only apply coloring if we have recited words to compare
        if i < len(recited_words):
            r = recited_words[i]
            similarity = calculate_similarity(e, r)
            if similarity >= accuracy_threshold:
                word_style = "color: green;"
                accuracy_count += 1
            else:
                word_style = "color: red;"
                error_details.append({
                    "position": i,
                    "expected": e,
                    "recited": r,
                    "similarity": similarity
                })
        
        # Always apply underline to current word
        if current_word_index is not None and i == current_word_index:
            word_style += " border-bottom: 2px solid #0c4b33;"
        
        if word_style:
            highlighted.append(f"<span style='{word_style}'>{e}</span>")
        else:
            highlighted.append(e)

    return " ".join(highlighted), accuracy_count, error_details

def recognize_generator(surah_num):
    state.update({
        "surah": int(surah_num),
        "ayah": 1,
        "buffer": "",
        "running": True,
        "recited_ayahs": {},
        "stop_requested": False,
        "current_attempt": {},
        "partial_result": "",
        "last_update": "",
        "surah_content": "",
        "errors": defaultdict(list),
        "recited_text": defaultdict(str),
        "expected_text": defaultdict(str),
        "completed_surahs": [],  # Track completed surahs
        "partial_ayah_buffer": ""  # New buffer to track partial ayah recitation
    })

    # Initial display with first word highlighted
    initial_display = display_surah_content(state["surah"], show_title=False, highlight_current_word=0)
    state["last_update"] = initial_display
    yield initial_display

    try:
        with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                             channels=1, callback=audio_callback):
            while state["running"]:
                data = q.get()
                
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text = result.get("text", "").strip()
                    if text:
                        state["buffer"] += " " + text
                        state["buffer"] = state["buffer"].strip()
                
                # Process partial results
                partial_result = json.loads(rec.PartialResult())
                partial_text = partial_result.get("partial", "").strip()
                
                if partial_text:
                    state["partial_result"] = partial_text
                    partial_words = partial_text.split()
                    remaining_buffer = partial_words.copy()
                    current_attempt = {}
                    
                    ayah_num = state["ayah"]
                    ayah_text = get_ayah(state["surah"], ayah_num)
                    
                    # Combine with any previously partially recited words
                    if state["partial_ayah_buffer"]:
                        partial_words = (state["partial_ayah_buffer"] + " " + partial_text).split()
                        remaining_buffer = partial_words.copy()
                    
                    while ayah_text and remaining_buffer:
                        ayah_words = ayah_text.split()
                        ayah_part = remaining_buffer[:len(ayah_words)]
                        remaining_buffer = remaining_buffer[len(ayah_words):]
                        
                        # Calculate current word position
                        current_word_pos = len(ayah_part) - 1 if ayah_part else 0
                        
                        highlighted, _, _ = highlight_words(ayah_text, " ".join(ayah_part), 
                                               current_word_index=current_word_pos)
                        
                        current_attempt[ayah_num] = {
                            "text": " ".join(ayah_part),
                            "highlighted": highlighted,
                            "current_word_pos": current_word_pos
                        }
                        
                        ayah_num += 1
                        ayah_text = get_ayah(state["surah"], ayah_num)
                    
                    state["current_attempt"] = current_attempt
                
                # Check for backward jumps
                buffer_words = state["buffer"].split()
                backward_jump_detected = False
                
                for previous_ayah in range(1, state["ayah"]):
                    prev_text = get_ayah(state["surah"], previous_ayah)
                    if not prev_text:
                        continue
                        
                    prev_words = prev_text.split()
                    if len(buffer_words) < len(prev_words):
                        continue
                        
                    test_sample = buffer_words[:len(prev_words)]
                    match_score = sum(calculate_similarity(e, r) >= accuracy_threshold 
                                    for e, r in zip(prev_words, test_sample)) / len(prev_words) * 100
                    
                    if match_score >= accuracy_threshold:
                        if previous_ayah == state["ayah"] - 1:
                            if state["ayah"] in state["recited_ayahs"]:
                                del state["recited_ayahs"][state["ayah"]]
                        else:
                            state["recited_ayahs"] = {
                                ayah_num: highlight 
                                for ayah_num, highlight in state["recited_ayahs"].items() 
                                if ayah_num <= previous_ayah
                            }
                            state["ayah"] = previous_ayah + 1
                        
                        buffer_words = buffer_words[len(prev_words):]
                        state["buffer"] = ' '.join(buffer_words).strip()
                        state["partial_ayah_buffer"] = ""  # Clear partial buffer on backward jump
                        backward_jump_detected = True
                        state["partial_result"] = ""
                        state["current_attempt"] = {}
                        break

                if not backward_jump_detected:
                    buffer_words = state["buffer"].split()
                    ayah_num = state["ayah"]
                    ayah_text = get_ayah(state["surah"], ayah_num)
                    surah_completed = False
                    
                    while ayah_text and len(buffer_words) >= len(ayah_text.split()):
                        ayah_words = ayah_text.split()
                        recited_part = buffer_words[:len(ayah_words)]
                        buffer_words = buffer_words[len(ayah_words):]
                        
                        accuracy = sum(calculate_similarity(e, r) >= accuracy_threshold 
                                 for e, r in zip(ayah_words, recited_part)) / len(ayah_words) * 100
                        
                        if accuracy >= 50:
                            highlighted, _, error_details = highlight_words(ayah_text, " ".join(recited_part))
                            state["recited_ayahs"][ayah_num] = highlighted
                            state["buffer"] = " ".join(buffer_words)
                            state["partial_ayah_buffer"] = ""  # Clear partial buffer on successful ayah completion
                            
                            # Store error details
                            state["errors"][ayah_num].extend(error_details)
                            state["recited_text"][ayah_num] = " ".join(recited_part)
                            state["expected_text"][ayah_num] = ayah_text
                            
                            ayah_num += 1
                            
                            # Check for surah completion
                            if ayah_num > max(quran.get(state["surah"], {}).keys()):
                                surah_completed = True
                                current_surah = state["surah"]
                                next_surah = current_surah + 1
                                
                                # Generate error report for completed surah
                                error_report = generate_error_report()
                                
                                # Keep only the most recent report
                                state["completed_surahs"] = [{
                                    "surah_num": current_surah,
                                    "report": error_report
                                }]
                                
                                if next_surah in quran:
                                    # Reset state for next surah
                                    state["surah"] = next_surah
                                    ayah_num = 1
                                    state["recited_ayahs"] = {}
                                    state["buffer"] = ""
                                    state["current_attempt"] = {}
                                    state["partial_result"] = ""
                                    state["errors"] = defaultdict(list)
                                    state["recited_text"] = defaultdict(str)
                                    state["expected_text"] = defaultdict(str)
                                    state["partial_ayah_buffer"] = ""
                                    
                                    # Display new surah first, then the error report below
                                    full_surah_html = display_surah_content(next_surah, show_title=False, highlight_current_word=0)
                                    combined_html = f"""
                                    <div class='new-surah-display'>
                                        <h3>Now Reciting: Surah {next_surah}</h3>
                                        {full_surah_html}
                                    </div>
                                    <div class='completed-surah-report'>
                                        <h3>Completed Surah {current_surah} Report</h3>
                                        {error_report}
                                    </div>
                                    """
                                    state["last_update"] = combined_html
                                    yield combined_html
                                    continue
                                else:
                                    state["running"] = False
                    
                    state["ayah"] = ayah_num
                    
                    # Handle partial ayah recitation (new logic)
                    if ayah_text and buffer_words:
                        ayah_words = ayah_text.split()
                        partial_match = False
                        
                        # Check if we have a partial match at the beginning of the ayah
                        if len(buffer_words) <= len(ayah_words):
                            partial_accuracy = sum(calculate_similarity(e, r) >= accuracy_threshold 
                                                for e, r in zip(ayah_words[:len(buffer_words)], buffer_words)) / len(buffer_words) * 100
                            
                            if partial_accuracy >= 50:
                                state["partial_ayah_buffer"] = " ".join(buffer_words)
                                partial_match = True
                        
                        if not partial_match:
                            state["partial_ayah_buffer"] = ""
                    
                    # If surah was completed but we're not moving to next surah (end of Quran)
                    if surah_completed and not state["running"]:
                        error_report = generate_error_report()
                        final_html = f"""
                        <div class='final-report'>
                            <h2>Recitation Complete</h2>
                            <div class='final-surah-report'>
                                {error_report}
                            </div>
                        </div>
                        """
                        state["last_update"] = final_html
                        yield final_html
                        return
                
                # Build display
                full_surah_html = display_surah_content(
                    state["surah"], 
                    show_title=False,
                    highlight_current_word=state["ayah"] if not state["current_attempt"] else None
                )
                
                # Apply highlighting to completed ayahs
                for ayah_num, highlighted in state["recited_ayahs"].items():
                    ayah_text = get_ayah(state["surah"], ayah_num)
                    ayah_html = f"{highlighted}<sup style='font-size:0.7em;'>۝</sup>"
                    full_surah_html = full_surah_html.replace(
                        f"{ayah_text}<sup style='font-size:0.7em;'>۝</sup>",
                        ayah_html
                    )
                
                # Apply real-time highlighting with underlines
                for ayah_num, attempt in state["current_attempt"].items():
                    ayah_text = get_ayah(state["surah"], ayah_num)
                    if ayah_text:
                        # Highlight the current word being recited
                        ayah_html = f"{attempt['highlighted']}<sup style='font-size:0.7em;'>۝</sup>"
                        full_surah_html = full_surah_html.replace(
                            f"{ayah_text}<sup style='font-size:0.7em;'>۝</sup>",
                            ayah_html
                        )
                
                # Add completed surah report if it exists
                current_display = full_surah_html
                if state["completed_surahs"]:
                    report = state["completed_surahs"][-1]  # Get the most recent report
                    current_display = f"""
                    <div class='current-recitation'>
                        {full_surah_html}
                    </div>
                    <div class='completed-surah-report'>
                        <h3>Completed Surah {report['surah_num']} Report</h3>
                        {report['report']}
                    </div>
                    """
                
                if current_display != state["last_update"] and not state["stop_requested"]:
                    state["last_update"] = current_display
                    yield current_display
                
                # Short sleep to prevent overwhelming the UI
                time.sleep(0.1)
                
    except Exception as e:
        yield f"<div class='error'>Error: {e}</div>"
    
    # After stopping, don't yield anything else
    if state["stop_requested"]:
        yield state["last_update"]

def stop_recitation():
    state["running"] = False
    state["stop_requested"] = True
    
    # Generate error report
    error_report = generate_error_report()
    
    # Get the current display without any further updates
    current_display = state["last_update"]
    
    # Combine with the error report in a way that won't be overwritten
    full_report = f"""
    <div class='recitation-report'>
        <div class='current-display'>
            {current_display}
        </div>
        <div class='error-analysis'>
            <h3>Recitation Analysis</h3>
            {error_report}
        </div>
    </div>
    """
    
    # Update the last display state to include the error report
    state["last_update"] = full_report
    
    return full_report

def generate_error_report():
    if not state["errors"]:
        return "<p class='success-message'>Excellent recitation! No errors detected.</p>"
    
    report_html = []
    total_errors = 0
    
    for ayah_num in sorted(state["errors"].keys()):
        errors = state["errors"][ayah_num]
        if not errors:
            continue
            
        total_errors += len(errors)
        expected_text = state["expected_text"][ayah_num]  # Original with full diacritics
        recited_text = state["recited_text"][ayah_num]    # What was actually recited
        
        # Create a detailed comparison with full diacritics
        comparison_html = []
        expected_words = expected_text.split()
        recited_words = recited_text.split()
        
        for i, expected_word in enumerate(expected_words):
            error_found = False
            for error in errors:
                if error["position"] == i:
                    # Get the actual recited word with harakat if available
                    recited_word_with_harakat = ""
                    if i < len(recited_words):
                        # Try to preserve harakat from original text where possible
                        original_word = expected_words[i]
                        recited_word = recited_words[i]
                        
                        # Create a hybrid word that shows the recited letters with original harakat
                        hybrid_word = []
                        original_chars = list(original_word)
                        recited_chars = list(recited_word)
                        
                        for oc in original_chars:
                            if oc in ['َ', 'ِ', 'ُ', 'ً', 'ٍ', 'ٌ', 'ْ', 'ّ']:
                                hybrid_word.append(oc)
                            elif recited_chars:
                                hybrid_word.append(recited_chars.pop(0))
                        
                        # Add any remaining recited characters
                        hybrid_word.extend(recited_chars)
                        recited_word_with_harakat = ''.join(hybrid_word)
                    else:
                        recited_word_with_harakat = "[Missing]"
                    
                    comparison_html.append(
                        f"<tr>"
                        f"<td class='expected-word'>{expected_word}</td>"
                        f"<td class='recited-word error-word'>{recited_word_with_harakat}</td>"
                        f"<td class='similarity'>{error['similarity']}%</td>"
                        f"</tr>"
                    )
                    error_found = True
                    break
            
            if not error_found:
                if i < len(recited_words):
                    # For correct words, show the original harakat
                    comparison_html.append(
                        f"<tr>"
                        f"<td class='expected-word'>{expected_word}</td>"
                        f"<td class='recited-word correct-word'>{expected_words[i]}</td>"
                        f"<td class='similarity'>100%</td>"
                        f"</tr>"
                    )
                else:
                    comparison_html.append(
                        f"<tr>"
                        f"<td class='expected-word'>{expected_word}</td>"
                        f"<td class='recited-word missing-word'>[Missing]</td>"
                        f"<td class='similarity'>0%</td>"
                        f"</tr>"
                    )
        
        report_html.append(f"""
        <div class='ayah-error-report'>
            <h4>Ayah {ayah_num} Errors ({len(errors)} errors)</h4>
            <div class='comparison-table'>
                <table>
                    <thead>
                        <tr>
                            <th>Expected</th>
                            <th>Your Recitation</th>
                            <th>Similarity</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(comparison_html)}
                    </tbody>
                </table>
            </div>
        </div>
        """)

    return "".join(report_html)

def display_surah_content(surah_num, show_title=True, highlight_current_word=None):
    if not surah_num:
        return ""

    ayah_list = []
    for ayah_num in sorted(quran.get(surah_num, {}).keys()):
        ayah_text = get_ayah(surah_num, ayah_num)
        
        # Highlight current word if specified
        if highlight_current_word is not None and ayah_num == highlight_current_word:
            words = ayah_text.split()
            if words:
                words[0] = f"<span class='current-word-highlight'>{words[0]}</span>"
                ayah_text = " ".join(words)
        
        ayah_html = f"{ayah_text}<sup style='font-size:0.7em;'>۝</sup>"
        ayah_list.append(ayah_html)

    surah_content = " ".join(ayah_list)
    
    if show_title:
        surah_name_ar = surah_names.get(surah_num, {}).get("ar", f"سورة {surah_num}")
        return f"""
        <div class='surah-display-panel'>
            <div class='surah-title'>
                <h2>Surah {surah_num}</h2>
                <div class='arabic'>{surah_name_ar}</div>
            </div>
            <div class='surah-content'>
                {surah_content}
            </div>
        </div>
        """
    else:
        return f"""
        <div class='surah-content'>
            {surah_content}
        </div>
        """

def filter_surahs(search_term):
    if not search_term:
        return [gr.update(visible=True) for _ in range(114)]
    
    try:
        search_num = int(search_term)
        return [gr.update(visible=(i+1 == search_num)) for i in range(114)]
    except ValueError:
        return [gr.update(visible=True) for _ in range(114)]

with gr.Blocks(css="""
.header {
    background: linear-gradient(135deg, #0c4b33 0%, #1a936f 100%);
    padding: 1.8rem 1.5rem;
    text-align: center;
    margin-bottom: 1rem;
    border-radius: 0 0 15px 15px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    border-bottom: 5px solid #d4af37;
    margin-top:-27px;
}

.header h1 {
    color: white;
    margin: 0;
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
}

               /* Completed Surah Reports */
.completed-surah-report {
    background: white;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    border-left: 4px solid #1a936f;
}

.completed-surah-report h3 {
    color: #0c4b33;
    margin-top: 0;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid rgba(26, 147, 111, 0.2);
}

.reports-container {
    margin-bottom: 2rem;
}

.current-recitation {
    margin-top: 1.5rem;
}

.final-report {
    background: white;
    padding: 2rem;
    border-radius: 15px;
    max-width: 900px;
    margin: 0 auto;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
}

.final-report h2 {
    color: #0c4b33;
    text-align: center;
    margin-bottom: 1.5rem;
}

.final-surah-report {
    padding: 1.5rem;
    background: #f9f9f9;
    border-radius: 10px;
}
.header p {
    color: white;
    margin: 1rem 0 0;
    font-size: 1.2rem;
    opacity: 0.9;
    font-weight: 300;
}

/* Enhanced Search Container */
.search-container {
    max-width: 700px;
    margin: 0rem auto 3rem;
    padding: 0 1.5rem;
    position: relative;
    background: linear-gradient(135deg, rgba(241, 248, 246, 0.3) 0%, rgba(241, 248, 246, 0.1) 100%);
    border-radius: 50px;
    box-shadow: 
        0 4px 20px rgba(12, 75, 51, 0.08),
        inset 0 1px 1px rgba(255, 255, 255, 0.3);
    border: 1px solid rgba(26, 147, 111, 0.15);
}

/* Search Box Styling */
.search-box {
    width: 100%;
    padding: 1.1rem 1.5rem 1.1rem 3.5rem !important;
    border: none !important;
    border-radius: 50px !important;
    font-size: 1.1rem !important;
    background: white url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%231a936f' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='11' cy='11' r='8'%3E%3C/circle%3E%3Cline x1='21' y1='21' x2='16.65' y2='16.65'%3E%3C/line%3E%3C/svg%3E") no-repeat 1.2rem center !important;
    background-size: 1.3rem !important;
    transition: all 0.3s ease !important;
    color: #0c4b33 !important;
    box-shadow: none !important;
}

.search-box:focus {
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(26, 147, 111, 0.3) !important;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%230c4b33' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='11' cy='11' r='8'%3E%3C/circle%3E%3Cline x1='21' y1='21' x2='16.65' y2='16.65'%3E%3C/line%3E%3C/svg%3E"), linear-gradient(to right, white 95%, rgba(26, 147, 111, 0.05) 100%) !important;
}

.search-box::placeholder {
    color: #7a9c8e !important;
    opacity: 0.8 !important;
    font-weight: 300 !important;
}

/* Decorative Elements */
.search-container::before {
    content: '';
    position: absolute;
    top: -15px;
    left: 50%;
    transform: translateX(-50%);
    width: 120px;
    height: 15px;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 120 15'%3E%3Cpath fill='%231a936f' fill-opacity='0.2' d='M0,7.5 Q30,15 60,7.5 T120,7.5'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: center;
}

.search-container::after {
    position: absolute;
    bottom: -25px;
    left: 50%;
    transform: translateX(-50%);
    font-family: 'Scheherazade', serif;
    font-size: 1.3rem;
    color: rgba(12, 75, 51, 0.5);
}

/* Surah Grid with Islamic Motif */
.surah-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 25px;
    padding: 20px;
    max-width: 1300px;
    margin: 0 auto;
}

.surah-card {
    background: white;
    border-radius: 12px;
    padding: 25px 15px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.05);
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    cursor: pointer;
    text-align: center;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(0,0,0,0.05);
}

.surah-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 5px;
    background: linear-gradient(90deg, #0c4b33, #1a936f);
}

.surah-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 15px 30px rgba(0,0,0,0.1);
}

.surah-number {
    background: linear-gradient(135deg, #0c4b33 0%, #1a936f 100%);
    color: white;
    width: 48px;
    height: 48px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 15px;
    font-weight: bold;
    font-size: 1.3rem;
    box-shadow: 0 5px 15px rgba(12, 75, 51, 0.3);
}

.surah-name-arabic {
    font-family: 'Scheherazade', serif;
    font-size: 2rem;
    color: #0c4b33;
    direction: rtl;
    line-height: 1.4;
    margin-top: 10px;
    font-weight: 600;
}

/* Enhanced Recitation Page Styles */
.recitation-container {
    max-width: 1550px;
    margin: 0rem auto;
    padding: 15px;
    background: #f9f9f9;
    border-radius: 15px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    border: 1px solid #e0e0e0;
}

/* Surah Display Panel with Islamic Art */
.surah-display-panel {
    background: white;
    border-radius: 15px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    padding: 3rem;
    margin: 0 auto;
    border: 1px solid #e0e0e0;
    border-top: 5px solid #0c4b33;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}

.surah-display-panel:hover {
    box-shadow: 0 15px 40px rgba(12, 75, 51, 0.15);
    transform: translateY(-2px);
}

.surah-display-panel::before {
    content: '';
    position: absolute;
    top: 0;
    right: 0;
    width: 150px;
    height: 150px;
    background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Cpath fill='%23f1f8f6' d='M50,0 C77.6,0 100,22.4 100,50 C100,77.6 77.6,100 50,100 C22.4,100 0,77.6 0,50 C0,22.4 22.4,0 50,0 Z'/%3E%3C/svg%3E") no-repeat;
    opacity: 0.2;
    z-index: 0;
}

.surah-display-panel::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100px;
    height: 100px;
    background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Cpath fill='%23f1f8f6' d='M50,0 C77.6,0 100,22.4 100,50 C100,77.6 77.6,100 50,100 C22.4,100 0,77.6 0,50 C0,22.4 22.4,0 50,0 Z'/%3E%3C/svg%3E") no-repeat;
    opacity: 0.2;
    z-index: 0;
    transform: rotate(180deg);
}

/* Surah Title with Decorative Border */
.surah-title {
    text-align: center;
    margin-bottom: 2.5rem;
    position: relative;
    z-index: 1;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid rgba(26, 147, 111, 0.3);
}

.surah-title h2 {
    font-size: 2.2rem;
    margin: 0;
    color: #0c4b33;
    font-weight: 600;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    position: relative;
    display: inline-block;
}

.surah-title h2::after {
    content: '';
    position: absolute;
    bottom: -15px;
    left: 50%;
    transform: translateX(-50%);
    width: 80px;
    height: 3px;
    background: linear-gradient(90deg, #0c4b33, #1a936f);
    border-radius: 3px;
}

.surah-title .arabic {
    font-family: 'Scheherazade', serif;
    font-size: 3rem;
    direction: rtl;
    color: #1a936f;
    margin-top: 10px;
    font-weight: 700;
    letter-spacing: 1px;
}

/* Enhanced Surah Content */
.surah-content {
    font-family: 'Scheherazade', serif;
    font-size: 1.8rem;
    line-height: 3;
    direction: rtl;
    text-align: right;
    padding: 1rem;
    background-color: rgba(241, 248, 246, 0.5);
    border-radius: 10px;
    position: relative;
    z-index: 1;
    border: 1px solid rgba(0,0,0,0.05);
    min-height: 300px;
    box-shadow: 
        inset 0 0 10px rgba(0,0,0,0.05),
        0 4px 6px rgba(0,0,0,0.01);
    margin-top: -10px;
    transition: all 0.3s ease;
}

.surah-content:hover {
    box-shadow: 
        inset 0 0 15px rgba(12, 75, 51, 0.1),
        0 6px 12px rgba(0,0,0,0.05);
}

/* Current Word Highlight Style */
.current-word-highlight {
    background-color: black !important;
    color: white !important;
    padding: 4px !important;
    border-radius: 3px;
    transition: all 0.2s ease;
    
}

/* Enhanced Microphone Button */
.mic-button {
    position: fixed;
    bottom: 30px;
    right: 30px;
    width: 70px;
    height: 70px;
    border-radius: 50%;
    background: linear-gradient(135deg, #0c4b33 0%, #1a936f 100%);
    color: white;
    border: none;
    box-shadow: 0 5px 20px rgba(0,0,0,0.2);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.8rem;
    transition: all 0.3s ease;
    z-index: 1000;
    border: 3px solid white;
}

.mic-button:hover {
    transform: scale(1.1);
    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
}

.mic-button.active {
    animation: pulse 1.5s infinite;
    background: linear-gradient(135deg, #d32f2f 0%, #f44336 100%);
    border-color: rgba(255,255,255,0.8);
}

/* Decorative Corner Elements */
.recitation-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 50px;
    height: 50px;
    border-top: 3px solid #0c4b33;
    border-left: 3px solid #0c4b33;
    border-radius: 15px 0 0 0;
}

.recitation-container::after {
    content: '';
    position: absolute;
    bottom: 0;
    right: 0;
    width: 50px;
    height: 50px;
    border-bottom: 3px solid #1a936f;
    border-right: 3px solid #1a936f;
    border-radius: 0 0 15px 0;
}

/* Current Word Underline Animation */
@keyframes current-word-underline {
    0% { border-bottom-color: rgba(12, 75, 51, 0.3); }
    50% { border-bottom-color: rgba(12, 75, 51, 0.8); }
    100% { border-bottom-color: rgba(12, 75, 51, 0.3); }
}

.current-word {
    border-bottom: 2px solid #0c4b33;
    animation: current-word-underline 1.5s infinite;
}

/* Pulse animation for microphone */
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(211, 47, 47, 0.7); }
    70% { box-shadow: 0 0 0 15px rgba(211, 47, 47, 0); }
    100% { box-shadow: 0 0 0 0 rgba(211, 47, 47, 0); }
}

/* Responsive Adjustments for Recitation Page */
@media (max-width: 768px) {
    .surah-display-panel {
        padding: 1.5rem;
        margin: 1rem auto;
    }
    
    .surah-content {
        font-size: 1.8rem;
        padding: 1rem;
        min-height: 200px;
    }
    
    .surah-title h2 {
        font-size: 1.8rem;
    }
    
    .surah-title .arabic {
        font-size: 2.2rem;
    }
    
    .mic-button {
        width: 60px;
        height: 60px;
        font-size: 1.5rem;
    }
    
    .recitation-container {
        padding: 15px;
    }
}

@media (max-width: 480px) {
    .surah-content {
        font-size: 1.5rem;
        line-height: 2.5;
    }
    
    .surah-title h2 {
        font-size: 1.5rem;
    }
    
    .surah-title .arabic {
        font-size: 1.8rem;
    }
    
    .recitation-container::before,
    .recitation-container::after {
        width: 30px;
        height: 30px;
    }
}

/* Fullscreen Splash Screen */
.splash-image-container {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    margin: 0 !important;
    padding: 0 !important;
    z-index: 9999 !important;
    overflow: hidden !important;
}

#splash-image {
    width: 100% !important;
    height: 100% !important;
    object-fit: cover !important;
    display: block !important;
    background:linear-gradient(135deg, #0c4b33 0%, #1a936f 100%);  
}
/* [Previous CSS remains the same until the end] */

/* Error Report Styles - Enhanced */
.recitation-report {
    background: white;
    padding: 1.5rem;
    border-radius: 20px;
    margin: 2rem auto;
    max-width: 1200px;
    box-shadow: 
        0 10px 30px rgba(12, 75, 51, 0.1),
        0 2px 10px rgba(0,0,0,0.05);
    border: 1px solid rgba(26, 147, 111, 0.15);
    position: relative;
    overflow: hidden;
}

.recitation-report::before {
    content: '';
    position: absolute;
    top: 0;
    right: 0;
    width: 150px;
    height: 150px;
    background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Cpath fill='%23f1f8f6' d='M50,0 C77.6,0 100,22.4 100,50 C100,77.6 77.6,100 50,100 C22.4,100 0,77.6 0,50 C0,22.4 22.4,0 50,0 Z'/%3E%3C/svg%3E") no-repeat;
    opacity: 0.3;
    z-index: 0;
}

.error-analysis {
    margin-top: 2rem;
    padding: 2rem;
    background: linear-gradient(to bottom, #f9f9f9, #f1f8f6);
    border-radius: 15px;
    border-left: 6px solid #d32f2f;
    box-shadow: inset 0 0 15px rgba(0,0,0,0.03);
    position: relative;
    z-index: 1;
}

.error-analysis h3 {
    font-size: 1.8rem;
    color: #0c4b33;
    margin-top: 0;
    padding-bottom: 1rem;
    border-bottom: 2px solid rgba(26, 147, 111, 0.2);
    font-weight: 600;
    display: flex;
    align-items: center;
}

.error-analysis h3::before {
    content: '📋';
    margin-right: 12px;
    font-size: 1.5rem;
}

.ayah-error-report {
    margin-bottom: 2.5rem;
    padding: 1.5rem;
    background: white;
    border-radius: 12px;
    box-shadow: 
        0 3px 10px rgba(0,0,0,0.05),
        inset 0 0 0 1px rgba(0,0,0,0.03);
    transition: all 0.3s ease;
    border-top: 3px solid #f1f8f6;
}

.ayah-error-report:hover {
    transform: translateY(-3px);
    box-shadow: 
        0 8px 20px rgba(12, 75, 51, 0.1),
        inset 0 0 0 1px rgba(26, 147, 111, 0.1);
}

.ayah-error-report h4 {
    color: #0c4b33;
    margin-top: 0;
    padding-bottom: 0.8rem;
    border-bottom: 1px dashed rgba(26, 147, 111, 0.3);
    font-size: 1.4rem;
    display: flex;
    align-items: center;
}

.ayah-error-report h4::before {
    content: '🖊️';
    margin-right: 10px;
    opacity: 0.7;
}

.comparison-table {
    overflow-x: auto;
    margin-top: 1.5rem;
    border-radius: 10px;
    background: white;
    box-shadow: 0 2px 5px rgba(0,0,0,0.02);
}

.comparison-table table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 10px;
    overflow: hidden;
}

.comparison-table th {
    background: linear-gradient(to right, #0c4b33, #1a936f);
    padding: 1rem;
    text-align: left;
    font-weight: 500;
    color: white;
    font-size: 1.1rem;
    position: sticky;
    top: 0;
}

.comparison-table th:first-child {
    border-top-left-radius: 8px;
}

.comparison-table th:last-child {
    border-top-right-radius: 8px;
}

.comparison-table td {
    padding-left:0.5rem;
    border-bottom: 1px solid rgba(0,0,0,0.05);
    vertical-align: middle;
    transition: all 0.2s ease;
}

.comparison-table tr:hover td {
    background: rgba(241, 248, 246, 0.5);
}

.comparison-table tr:last-child td {
    border-bottom: none;
}

.expected-word {
    font-family: 'Scheherazade', serif;
    font-size: 1.6rem;
    color: #0c4b33;
    direction: rtl;
    line-height: 1.8;
}

.recited-word {
    font-family: 'Scheherazade', serif;
    font-size: 1.6rem;
    direction: rtl;
    line-height: 1.8;
}

.correct-word {
    color: #2e7d32;
    position: relative;
}

.correct-word::after {
    content: '✓';
    margin-left: 8px;
    color: #2e7d32;
    font-size: 1rem;
}

.error-word {
    color: #d32f2f;
    font-weight: bold;
    position: relative;
}

.error-word::after {
    content: '✗';
    margin-left: 8px;
    color: #d32f2f;
    font-size: 1rem;
}

.missing-word {
    color: #666;
    font-style: italic;
    position: relative;
}

.missing-word::before {
    content: '→';
    margin-right: 8px;
    color: #666;
}

.similarity {
    font-family: 'Courier New', monospace;
    text-align: center;
    font-weight: bold;
    font-size: 1.1rem;
    min-width: 80px;
}


.success-message {
    color: #2e7d32;
    font-size: 1.3rem;
    padding: 1.5rem;
    background: rgba(46, 125, 50, 0.1);
    border-radius: 10px;
    text-align: center;
    border: 1px solid rgba(46, 125, 50, 0.2);
    margin: 1rem 0;
    display: flex;
    align-items: center;
    justify-content: center;
}

.success-message::before {
    content: '✓';
    margin-right: 12px;
    font-size: 1.8rem;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .recitation-report {
        padding: 1.5rem;
    }
    
    .error-analysis {
        padding: 1.5rem;
    }
    
    .ayah-error-report {
        padding: 1rem;
    }
    
    .comparison-table th,
    .comparison-table td {
        padding: 0rem;
    }
    
    .expected-word,
    .recited-word {
        font-size: 1.4rem;
    }
}

@media (max-width: 480px) {
    .recitation-report {
        padding: 0rem;
    }
    
    .error-analysis {
        padding: 1rem;
    }
    
    .error-summary {
        padding: 1rem;
    }
    
    .comparison-table th {
        font-size: 0.9rem;
        padding: 0.2rem;
    }
    
    .comparison-table td {
        padding: 0rem;
    }
    
    .expected-word,
    .recited-word {
        font-size: 1.2rem;
    }
}
""") as app:
    
    with gr.Group(elem_classes="splash-image-container") as splash_group:
        gr.Image("E:/FYP/4.png", elem_id="splash-image", 
                show_label=False, show_download_button=False)

    # Enhanced Home Page Only
    with gr.Tab("Home", visible=False) as main_group:
        with gr.Column(elem_classes="home-page"):
            # Elegant Islamic-themed Header
            with gr.Row(elem_classes="header"):
                gr.HTML("""
                <div>
                    <h1>Al-Hafiz Companion</h1>
                    <p>The Ultimate Hifz Learning System</p>
                </div>
                """)
            
            # Decorative Search Box
            with gr.Row(elem_classes="search-container"):
                search_box = gr.Textbox(placeholder="Search by Surah number...", 
                                      elem_classes="search-box")
            
            # Beautiful Surah Grid
            with gr.Row():
                with gr.Column(elem_classes="surah-grid"):
                    selected_surah = gr.Number(value=1, visible=False)
                    
                    surah_cards = []
                    for surah_num in range(1, 115):
                        surah_name_ar = surah_names.get(surah_num, {}).get("ar", f"سورة {surah_num}")
                        card = gr.HTML(
                            f"""
                            <div class="surah-card" onclick="this.dispatchEvent(new Event('click'))">
                                <div class="surah-number">{surah_num}</div>
                                <div class="surah-name-arabic">{surah_name_ar}</div>
                            </div>
                            """,
                            visible=True
                        )
                        card.click(fn=lambda x=surah_num: x, outputs=selected_surah)
                        surah_cards.append(card)
            
            search_box.change(fn=filter_surahs, inputs=search_box, outputs=surah_cards)

    # Real-time Recitation tab
    with gr.Tab("Real-time Recitation"):
        with gr.Column(elem_classes="recitation-container"):
            # Surah display
            with gr.Row():
                with gr.Column():
                    surah_content_display = gr.HTML()
        
        # Floating microphone button
        mic_button = gr.Button("🎤", elem_classes="mic-button", elem_id="stop-button")

        # Event handlers
        mic_button.click(
            fn=lambda: (stop_recitation(), None),
            outputs=[surah_content_display, gr.Textbox(visible=False)]
        )
        mic_button.click(
            recognize_generator, 
            inputs=selected_surah, 
            outputs=surah_content_display
        )
        selected_surah.change(
            lambda x: display_surah_content(x, show_title=False, highlight_current_word=1),
            inputs=selected_surah, 
            outputs=surah_content_display
        )

    def show_main():
        time.sleep(3)
        return gr.update(visible=False), gr.update(visible=True)
    
    app.load(show_main, outputs=[splash_group, main_group])

app.launch()