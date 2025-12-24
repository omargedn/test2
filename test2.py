#!/usr/bin/env python3
"""
Real Estate Lead Automation System
Streamlined Version - Single Property Only
(No-Spacy Version for compatibility)
"""
import time
from datetime import datetime
import json 
from openai import OpenAI
import os
import re
import tempfile
import warnings
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import requests
import streamlit as st
import time

warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def local_css():
    st.markdown("""
    <style>
        /* Enhanced UI Styling */
        .stTextArea textarea {font-family: 'Courier New', monospace; font-size: 14px;}
        .metric-card {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        }
        .highlight { color: #00cc00; font-weight: bold; }
        .warning { color: #ffa500; font-weight: bold; }
        .danger { color: #ff4b4b; font-weight: bold; }
        
        /* Progress Bar Styling */
        .progress-container {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
        }
        .progress-step {
            display: flex;
            align-items: center;
            margin: 8px 0;
            padding: 8px;
            border-radius: 5px;
        }
        .progress-step.active {
            background-color: #e3f2fd;
            border-left: 4px solid #2196F3;
        }
        .progress-step.complete {
            background-color: #e8f5e9;
            border-left: 4px solid #4CAF50;
        }
        .progress-step.error {
            background-color: #ffebee;
            border-left: 4px solid #f44336;
        }
        
        /* Validation Warnings */
        .validation-warning {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px;
            margin: 5px 0;
            border-radius: 4px;
        }
        .validation-error {
            background-color: #f8d7da;
            border-left: 4px solid #dc3545;
            padding: 10px;
            margin: 5px 0;
            border-radius: 4px;
        }
        .validation-success {
            background-color: #d4edda;
            border-left: 4px solid #28a745;
            padding: 10px;
            margin: 5px 0;
            border-radius: 4px;
        }
        
        /* Confidence Indicators */
        .confidence-high { color: #28a745; font-weight: bold; }
        .confidence-medium { color: #ffc107; font-weight: bold; }
        .confidence-low { color: #dc3545; font-weight: bold; }
        
        /* Audio Player Styling */
        .audio-player-container {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        
        /* Enhanced Cards */
        .info-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .success-card {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .warning-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        
        /* Main container improvements */
        .main-container {
            padding: 20px;
        }
        
        /* Better button styling */
        .stButton > button {
            border-radius: 8px;
            transition: all 0.3s;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
    </style>
    """, unsafe_allow_html=True)

def render_audio_player(audio_url: str, transcript: str = None):
    """Render an HTML5 audio player with transcript synchronization"""
    if not audio_url or 'http' not in audio_url:
        st.warning("⚠️ No valid audio URL provided")
        return
    
    # Create a unique ID for this audio player
    player_id = f"audio_player_{hash(audio_url) % 10000}"
    
    audio_html = f"""
    <div class="audio-player-container">
        <h4>🎵 Call Recording</h4>
        <audio id="{player_id}" controls style="width: 100%; margin: 10px 0;">
            <source src="{audio_url}" type="audio/mpeg">
            Your browser does not support the audio element.
        </audio>
        <div style="margin-top: 10px;">
            <small>💡 Tip: Use the player controls to play, pause, and adjust volume</small>
        </div>
    </div>
    """
    
    st.markdown(audio_html, unsafe_allow_html=True)
    
    # If transcript is provided, add search/highlight functionality
    if transcript:
        st.markdown("### 📝 Transcript Search")
        search_term = st.text_input("Search in transcript:", key=f"search_{player_id}")
        
        if search_term:
            # Highlight search terms in transcript
            highlighted_transcript = transcript.replace(
                search_term, 
                f"<mark style='background-color: yellow;'>{search_term}</mark>"
            )
            st.markdown(highlighted_transcript, unsafe_allow_html=True)
        else:
            st.text_area("Full Transcript", transcript, height=300, key=f"transcript_{player_id}")

# --- Dataclass ---
@dataclass
class FieldData:
    value: Any
    source: str
    confidence: float = 1.0



class ProcessStatus:
    def __init__(self):
        self.stages = {
            'file_upload': {'status': 'waiting', 'message': 'File Upload', 'progress': 0, 'estimated_time': 2},
            'data_parsing': {'status': 'waiting', 'message': 'Data Parsing', 'progress': 0, 'estimated_time': 3}, 
            'audio_transcription': {'status': 'waiting', 'message': 'Audio Transcription', 'progress': 0, 'estimated_time': 30},
            'ai_analysis': {'status': 'waiting', 'message': 'AI Analysis', 'progress': 0, 'estimated_time': 45},
            'qualification': {'status': 'waiting', 'message': 'Lead Qualification', 'progress': 0, 'estimated_time': 10},
            'report_generation': {'status': 'waiting', 'message': 'Report Generation', 'progress': 0, 'estimated_time': 5}
        }
        self.start_time = None
        self.current_stage = None
    
    def update_stage(self, stage, status, message=None, progress=None):
        self.stages[stage]['status'] = status
        if message:
            self.stages[stage]['message'] = message
        if progress is not None:
            self.stages[stage]['progress'] = progress
        if status == 'processing':
            self.current_stage = stage
            if self.start_time is None:
                self.start_time = time.time()
    
    def get_total_progress(self):
        """Calculate total progress percentage"""
        total_steps = len(self.stages)
        completed = sum(1 for s in self.stages.values() if s['status'] == 'complete')
        current = sum(1 for s in self.stages.values() if s['status'] == 'processing')
        
        if current > 0:
            # Get progress of current stage
            current_stage = next((s for s in self.stages.values() if s['status'] == 'processing'), None)
            if current_stage:
                base_progress = (completed / total_steps) * 100
                current_progress = (current_stage.get('progress', 0) / 100) * (100 / total_steps)
                return min(base_progress + current_progress, 100)
        
        return (completed / total_steps) * 100
    
    def get_estimated_time_remaining(self):
        """Calculate estimated time remaining"""
        if not self.current_stage:
            return 0
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        remaining_stages = [s for s in self.stages.values() if s['status'] in ['waiting', 'processing']]
        
        # Estimate based on remaining stages
        total_remaining = sum(s.get('estimated_time', 0) for s in remaining_stages)
        
        # Adjust based on current progress
        if self.current_stage:
            current = self.stages[self.current_stage]
            if current['status'] == 'processing':
                progress = current.get('progress', 0) / 100
                current_remaining = current.get('estimated_time', 0) * (1 - progress)
                total_remaining = current_remaining + sum(
                    s.get('estimated_time', 0) 
                    for k, s in self.stages.items() 
                    if k != self.current_stage and s['status'] == 'waiting'
                )
        
        return max(0, int(total_remaining - elapsed))
    
    def display_enhanced_status(self, progress_container=None):
        """Display enhanced progress with bars and time estimates"""
        total_progress = self.get_total_progress()
        time_remaining = self.get_estimated_time_remaining()
        
        if progress_container:
            with progress_container:
                # Overall progress bar
                st.progress(total_progress / 100)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Overall Progress", f"{total_progress:.1f}%")
                with col2:
                    if time_remaining > 0:
                        st.metric("Est. Time Remaining", f"{time_remaining}s")
                    else:
                        st.metric("Est. Time Remaining", "Calculating...")
                
                st.divider()
                
                # Step-by-step status
                for stage_key, info in self.stages.items():
                    status = info['status']
                    message = info['message']
                    progress = info.get('progress', 0)
                    
                    if status == 'processing':
                        icon = '🔄'
                        status_class = 'active'
                        st.markdown(f"<div class='progress-step {status_class}'>", unsafe_allow_html=True)
                        st.write(f"{icon} **{message}** (In Progress...)")
                        if progress > 0:
                            st.progress(progress / 100)
                        st.markdown("</div>", unsafe_allow_html=True)
                    elif status == 'complete':
                        icon = '✅'
                        status_class = 'complete'
                        st.markdown(f"<div class='progress-step {status_class}'>", unsafe_allow_html=True)
                        st.write(f"{icon} **{message}** (Complete)")
                        st.markdown("</div>", unsafe_allow_html=True)
                    elif status == 'warning':
                        icon = '⚠️'
                        status_class = 'error'
                        st.markdown(f"<div class='progress-step {status_class}'>", unsafe_allow_html=True)
                        st.write(f"{icon} **{message}** (Warning)")
                        st.markdown("</div>", unsafe_allow_html=True)
                    elif status == 'failed':
                        icon = '❌'
                        status_class = 'error'
                        st.markdown(f"<div class='progress-step {status_class}'>", unsafe_allow_html=True)
                        st.write(f"{icon} **{message}** (Failed)")
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        icon = '⚪'
                        st.write(f"{icon} {message} (Waiting)")
    
    def display_status(self):
        """Legacy method for backward compatibility"""
        for stage, info in self.stages.items():
            status = info['status']
            if status == 'processing': icon = '🔄'
            elif status == 'complete': icon = '✅' 
            elif status == 'warning': icon = '⚠️'
            elif status == 'failed': icon = '❌'
            else: icon = '⚪'
            
            st.write(f"{icon} {info['message']}")

# --- NLPAnalyzer Class ---
class NLPAnalyzer:
    """Analyzes conversation transcripts using regex for motivation."""
    
    def __init__(self):
        pass

    def analyze_transcript(self, transcript: str) -> Dict[str, str]:
        """Run the NLP pipeline on a transcript."""
        if not transcript:
            return {
                'motivation': "No transcript provided",
                'highlights': "No transcript provided"  
            }

        transcript_lower = transcript.lower()
        motivation = self._analyze_motivation(transcript_lower)
        
        return {
            'motivation': motivation,
            'highlights': "To be analyzed by AI"
        }

    def _analyze_motivation(self, transcript_lower: str) -> str:
        """Contextually analyze motivation/timeline."""
        high_motivation_patterns = [
            r"asap", r"soon as possible", r"immediate(ly)?",
            r"urgent", r"quick(ly)?", r"fast", r"motivated", r"ready",
            r"i can sell it right now", r"need to sell", r"have to sell",
            r"time sensitive", r"deadline", r"quick sale", r"fast closing"
        ]
        if any(re.search(p, transcript_lower) for p in high_motivation_patterns):
            return "Highly motivated - wants quick sale"
        
        low_motivation_patterns = [
            r"flexible", r"no rush", r"whenever", r"listing it", 
            r"testing (the )?market", r"just looking", r"seeing what",
            r"not in a hurry", r"take your time", r"whenever you"
        ]
        if any(re.search(p, transcript_lower) for p in low_motivation_patterns):
            return "Flexible timeline / Testing market"
        
        moderate_patterns = [
            r"want to sell", r"looking to sell", r"interested in selling",
            r"considering offers", r"ready to move", r"planning to sell"
        ]
        if any(re.search(p, transcript_lower) for p in moderate_patterns):
            return "Moderately motivated - open to selling"
        
        return "No motivation details discussed"

# --- FormParser Class ---
class FormParser:
    """Parse real estate lead forms with enhanced field processing"""
    
    def __init__(self):
        self.field_patterns = {
            'list_name': ['List Name'],
            'property_type': ['Property Type'],
            'seller_name': ['Seller Name'],
            'phone_number': ['Phone Number'],
            'address': ['Address'],
            'zillow_link': ['Zillow link'],
            'asking_price': ['Asking Price'],
            'zillow_estimate': ['Zillow Estimate'],
            'realtor_estimate': ['Realtor Estimate'],
            'redfin_estimate': ['Redfin Estimate'],
            'reason_for_selling': ['Reason For Selling'],
            'motivation_details': ['Motivation details'],
            'mortgage': ['Mortgage'],
            'condition': ['Condition'],
            'occupancy': ['Occupancy'],
            'closing_time': ['Closing time'],
            'best_time_to_call': ['Best time to call back'],
            'agent_name': ['Agent Name'],
            'call_recording': ['Call recording']
        }
    
    def parse_file(self, file_path: str) -> Dict[str, FieldData]:
        """Parse lead file into structured data"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return self.parse_text(content)
    
    def parse_text(self, text: str) -> Dict[str, FieldData]:
        """Extract and clean field values from text"""
        data = {}
        
        for field, names in self.field_patterns.items():
            value = self._extract_field(text, names)
            if value:
                cleaned_value = self._clean_field(field, value)
                data[field] = FieldData(value=cleaned_value, source='form')
            else:
                data[field] = FieldData(value="", source='form', confidence=0.0)
        
        return data
    
    def _extract_field(self, text: str, field_names: List[str]) -> Optional[str]:
        """Extract field value using your specific format"""
        for name in field_names:
            patterns = [
                rf'◇{name}\s*:-\s*(.+)',
                rf'◇{name}\s*:\s*(.+)',
                rf'{name}\s*:-\s*(.+)',
                rf'{name}\s*:\s*(.+)',
            ]
            
            # Special handling for call recording - can have URL directly after field name without colon
            if 'call recording' in name.lower():
                patterns.extend([
                    rf'◇{name}\s+(https?://[^\s\n]+)',  # URL directly after field name (same line)
                    rf'{name}\s+(https?://[^\s\n]+)',   # Without ◇ symbol (same line)
                    rf'◇{name}\s*\n\s*(https?://[^\s\n]+)',  # URL on next line
                    rf'{name}\s*\n\s*(https?://[^\s\n]+)',   # Without ◇, URL on next line
                    rf'◇{name}\s+([^\n]+)',       # Any text after field name (fallback)
                    rf'{name}\s+([^\n]+)',        # Without ◇ symbol (fallback)
                ])
            
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
                if matches:
                    value = matches[-1].strip()
                    
                    if 'asking price' in name.lower():
                        if (not value or 
                            value.lower() in ['', 'not specified', 'n/a', 'na', 'not available', 'unknown', 'not mentioned', 'not provided', 'none', 'null'] or
                            any(phrase in value.lower() for phrase in ['not mentioned', 'not provided', 'none', 'null', 'not discussed', 'no price', 'no asking'])):
                            return "Waiting for our offer"
                        
                        if 'negotiable' in value.lower():
                            return value + " (negotiable)"
                    
                    if value and value not in ['', 'Not specified', 'N/A', 'n/a']:
                        value = re.sub(r'^◇.*Estimate\s*:-\s*', '', value)
                        value = re.sub(r'^[^a-zA-Z0-9$]*', '', value)
                        return value
        return None
    
    def _clean_field(self, field: str, value: str) -> str:
        """Clean and normalize field values"""
        cleaners = {
            'phone_number': self._clean_phone,
            'seller_name': self._clean_name,
            'agent_name': self._clean_name,
            'asking_price': self._clean_price,
            'zillow_estimate': self._clean_price,
            'realtor_estimate': self._clean_price,
            'redfin_estimate': self._clean_price,
            'property_type': self._clean_property_type,
            'best_time_to_call': self._clean_time,
            'occupancy': self._clean_occupancy,
            'mortgage': self._clean_mortgage,
            'condition': self._clean_condition,
            'reason_for_selling': self._clean_reason,
            'closing_time': self._clean_closing_time,
            'motivation_details': self._clean_motivation,
            'call_recording': self._clean_call_recording,
        }
        
        cleaner = cleaners.get(field, lambda x: x.strip())
        return cleaner(value)
    
    def _clean_phone(self, phone: str) -> str:
        """Normalize phone number format"""
        digits = re.sub(r'[^\d]', '', phone)
        if len(digits) == 11 and digits.startswith('1'):
            digits = digits[1:]
        if len(digits) == 10:
            return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        return phone
    
    def _clean_name(self, name: str) -> str:
        return name.strip().title()

    def _clean_price(self, price: str) -> str:
        """Enhanced price cleaner that preserves negotiable notation"""
        if not price: 
            return price
        
        price = price.strip()
        is_negotiable = "(negotiable)" in price
        base_price = price.replace("(negotiable)", "").strip()
        
        if base_price == "Waiting for our offer":
            return base_price
        
        base_price = re.sub(r'^◇.*Estimate\s*:-\s*', '', base_price)
        base_price = re.sub(r'^[^a-zA-Z0-9$]*', '', base_price)
        
        if base_price.upper().endswith('K'):
            try:
                numeric_value = float(base_price.upper().replace('K', '').replace('$', '').replace(',', '').strip())
                cleaned_price = f"${numeric_value * 1000:,.0f}"
                return f"{cleaned_price} (negotiable)" if is_negotiable else cleaned_price
            except ValueError: 
                pass
        
        numbers = re.findall(r'([\d,]+\.?\d*)', base_price)
        if numbers:
            valid_numbers = []
            for num_str in numbers:
                clean_num_str = num_str.replace(',', '').replace('$', '').strip()
                if clean_num_str:
                    try:
                        numeric_value = float(clean_num_str)
                        valid_numbers.append((num_str, numeric_value))
                    except ValueError:
                        continue
            
            if valid_numbers:
                largest_num_str, largest_value = max(valid_numbers, key=lambda x: x[1])
                if largest_value == int(largest_value):
                    cleaned_price = f"${int(largest_value):,}"
                else:
                    cleaned_price = f"${largest_value:,.2f}"
                return f"{cleaned_price} (negotiable)" if is_negotiable else cleaned_price
        
        return price

    def _clean_property_type(self, prop_type: str) -> str:
        """Basic property type cleanup"""
        cleaned = prop_type.strip()
        cleaned = re.sub(r'^◇\s*Property\s*Type\s*:-\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'^◇\s*Property\s*Type\s*:\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def _clean_time(self, time: str) -> str:
        time = time.strip().lower()
        time_map = {
            'asap': 'As soon as possible',
            'anytime': 'Any time',
            'ay time': 'Any time',
        }
        return time_map.get(time, time.title())
    
    def _clean_occupancy(self, occupancy: str) -> str:
        """Enhanced occupancy cleaner"""
        occupancy_lower = occupancy.lower().strip()
        
        if occupancy_lower in ['n/a', 'na', 'not available', 'unknown', 'not specified', '']:
            return "Not specified"
        
        if 'vacant lot' in occupancy_lower:
            return "Vacant Lot"
        
        if any(phrase in occupancy_lower for phrase in [
            "30 day notice", "30-day notice", "30 days notice", 
            "notice to vacate", "submitted notice", "given notice"
        ]):
            return "Tenant Occupied (30-day notice given)"
        
        vacant_patterns = [
            'vacant', 'empty', 'no one living', 'nobody living', 
            'unoccupied', 'not occupied'
        ]
        if any(pattern in occupancy_lower for pattern in vacant_patterns):
            return "Vacant"
        
        tenant_patterns = [
            'tenant', 'rented', 'renting', 'occupied by tenant', 'renter',
            'currently rented', 'has a tenant', 'tenant occupied', 'lease'
        ]
        if any(pattern in occupancy_lower for pattern in tenant_patterns):
            return "Tenant Occupied"
        
        owner_patterns = [
            'owner occupied', 'primary residence', 'i live here', 'we live here',
            'owner-occupied', 'living in it', 'reside there'
        ]
        if any(pattern in occupancy_lower for pattern in owner_patterns):
            return "Owner Occupied"
        
        return occupancy.strip().capitalize()
    
    def _clean_mortgage(self, mortgage: str) -> str:
        """Enhanced mortgage cleaner"""
        mortgage_lower = mortgage.lower().strip()
        
        if mortgage_lower in ['n/a', 'na', 'not available', 'unknown', 'not specified', '']:
            return "Not available"
        
        if any(word in mortgage_lower for word in ['free and clear', 'own', 'paid off', 'no mortgage']):
            return "Owned free and clear"
        if any(word in mortgage_lower for word in ['mortgage', 'loan', 'yes', '$', 'left', 'owe']):
            return "Mortgage exists"
        return "Mortgage status unknown"
    
    def _clean_condition(self, condition: str) -> str:
        """
        Clean condition field but PRESERVE specific details.
        Do not replace detailed text with generic labels.
        """
        if not condition:
            return "Not specified"
            
        condition_lower = condition.lower().strip()
        
        # Only normalize if it's a useless placeholder
        if condition_lower in ['n/a', 'na', 'not available', 'unknown', 'not specified', '', 'none']:
            return "Not specified"
            
        # If it's just a generic word, maybe capitalize it, but don't change it too much
        # If the text is long (has details), return it as-is
        return condition.strip()
    
    def _clean_reason(self, reason: str) -> str:
        if not reason or not reason.strip():
            return "Not mentioned"
        
        reason_lower = reason.lower()
        if any(phrase in reason_lower for phrase in ['fix', 'investment', 'flip']):
            return "Property investment business"
        if "taxes" in reason_lower:
            return "Financial pressure from property taxes"
        if any(word in reason_lower for word in ['relocat', 'move']):
            return "Relocation"
        
        # If reason doesn't match specific patterns, return original text or "Not mentioned"
        # This avoids the generic "Standard property disposition" label
        return reason.strip() if reason.strip() else "Not mentioned"
    
    def _clean_closing_time(self, time: str) -> str:
        time = time.strip().lower()
        if 'asap' in time: return 'As soon as possible'
        return time.title()
    
    def _clean_motivation(self, motivation: str) -> str:
        motivation_lower = motivation.lower()
        if any(word in motivation_lower for word in ['high', 'very', 'urgent']):
            return "Highly motivated"
        if any(word in motivation_lower for word in ['motivated', 'ready']):
            return "Motivated"
        return "Standard motivation"
    
    def _clean_call_recording(self, url: str) -> str:
        """Clean and validate call recording URL"""
        if not url:
            return ""
        
        # Remove any leading/trailing whitespace and newlines
        url = url.strip()
        
        # Remove any trailing punctuation that might have been included
        url = url.rstrip('.,;:')
        
        # Ensure it's a valid URL format
        if url.startswith('http://') or url.startswith('https://'):
            return url
        
        # If it doesn't start with http, try to fix it
        if url.startswith('www.'):
            return 'https://' + url
        
        return url

# --- AudioProcessor Class ---
# --- AudioProcessor Class (Groq API Version) ---
class AudioProcessor:
    """
    Handle audio transcription using Groq API.
    Saves RAM by offloading processing to the cloud.
    """
    
    def __init__(self):
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        try:
            # Requires `pip install groq`
            from groq import Groq
            api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
            if not api_key:
                st.error("❌ GROQ_API_KEY not found. Please add it to secrets for audio transcription.")
                return None
            return Groq(api_key=api_key)
        except ImportError:
            st.error("❌ Groq library not found. Please add `groq` to your requirements.txt")
            return None
        except Exception as e:
            st.error(f"❌ Failed to initialize Groq client: {e}")
            return None

    def transcribe_audio(self, audio_url: str, status_tracker=None) -> Dict[str, Any]:
        """Transcribe audio using Groq's ultra-fast Whisper API"""
        
        if status_tracker:
            status_tracker.update_stage('audio_transcription', 'processing', 'Starting audio transcription...')
        
        result = {
            'transcript': None,
            'language': None, 
            'success': False,
            'error': None
        }

        if not self.client:
            result['error'] = "Groq Client not initialized"
            return result

        if not audio_url or 'http' not in audio_url:
            error_msg = "Invalid audio URL"
            result['error'] = error_msg
            if status_tracker:
                status_tracker.update_stage('audio_transcription', 'failed', error_msg)
            return result

        temp_path = None
        try:
            if status_tracker:
                status_tracker.update_stage('audio_transcription', 'processing', 'Downloading audio file...')
            
            # Download audio
            response = requests.get(audio_url, timeout=60)
            response.raise_for_status()

            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
                f.write(response.content)
                temp_path = f.name
            
            if status_tracker:
                status_tracker.update_stage('audio_transcription', 'processing', 'Sending to Groq API (Fast & 0 RAM)...')

            # --- GROQ API CALL ---
            with open(temp_path, "rb") as file:
                transcription = self.client.audio.transcriptions.create(
                    file=(temp_path, file.read()),
                    model="whisper-large-v3", # Use the best model
                    response_format="verbose_json",
                )
            # ---------------------

            result['transcript'] = transcription.text.strip()
            result['language'] = getattr(transcription, 'language', 'en')
            result['success'] = True
            
            if status_tracker:
                status_tracker.update_stage('audio_transcription', 'complete', 'Transcription completed successfully')
            
            st.success("✅ Transcription complete (Groq API)")

        except Exception as e:
            error_msg = str(e)
            result['error'] = error_msg
            
            if status_tracker:
                status_tracker.update_stage('audio_transcription', 'failed', f'Transcription failed: {error_msg}')
            
            st.error(f"❌ Transcription failed: {error_msg}")
            
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
                
        return result

# --- ConversationSummarizer Class ---
class ConversationSummarizer:
    """Generates a summary of the conversation in the new format."""
    
    def __init__(self, ai_client=None):
        self.client = ai_client
        self.model = "deepseek-chat"
    
    def summarize(self, transcript: str, nlp_data: Dict[str, str]) -> str:
        """Generate enhanced summary in the new format: Summary, Discussion Highlights, Action Items, Full Transcript"""
        if not transcript:
            return "No transcript available for summarization."
        
        # Use AI to generate the structured analysis
        if self.client:
            try:
                system_prompt = f"""
                You are an expert real estate call analyst. Analyze the following call transcript and create a structured analysis.

                CRITICAL FORMATTING RULES:
                1. Start with "Summary" as a header (no dashes, just the word "Summary")
                2. Write 2-3 sentences summarizing the overall purpose and outcome of the call.
                3. Add a blank line, then "Discussion Highlights" as a header
                4. List 5-8 key points discussed (each on a new line, no bullet points needed - just plain text with line breaks)
                5. Add a blank line, then "Action Items" as a header
                6. List specific next steps or commitments made (each on a new line, no bullet points)
                7. Be specific and factual - use information directly from the transcript.
                8. Format exactly like this example:
                
                Summary
                [2-3 sentences about the call]
                
                Discussion Highlights
                [Point 1]
                [Point 2]
                [Point 3]
                
                Action Items
                [Action 1]
                [Action 2]
                
                Transcript:
                {transcript}
                """
                
                chat_completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": "Analyze this call and create the structured summary with Summary, Discussion Highlights, and Action Items."}
                    ],
                    temperature=0.1,
                    max_tokens=800
                )
                
                ai_summary = chat_completion.choices[0].message.content.strip()
                return ai_summary
                
            except Exception as e:
                st.error(f"❌ AI Summary Error: {e}")
                # Fallback to basic format
                return self._generate_fallback_summary(transcript, nlp_data)
        else:
            # Fallback if no AI client
            return self._generate_fallback_summary(transcript, nlp_data)
    
    def _generate_fallback_summary(self, transcript: str, nlp_data: Dict[str, str]) -> str:
        """Fallback summary if AI is not available"""
        summary_lines = [
            "Summary",
            "Agent called to inquire about purchasing the property.",
            "",
            "Discussion Highlights",
            "• Agent expressed interest in buying the property",
            "• Property details were discussed",
            "• Closing timeline and reasons for selling were discussed",
            "",
            "Action Items",
            "• Agent to review information and follow up",
            ""
        ]
        return "\n".join(summary_lines)

# --- AIRephraser Class ---
class AIRephraser:
    """Complete AI analysis for all major conversation topics"""
    
    def __init__(self):
        self.client = None
        self.model = "deepseek-chat"
        self._initialize_client()
    
    def _initialize_client(self):
        self.client = self._get_cached_client()
    
    @st.cache_resource
    def _get_cached_client(_self):
        try:
            api_key = st.secrets["DEEPSEEK_API_KEY"]
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com/v1"
            )
            return client
        except Exception as e:
            st.error(f"❌ Failed to initialize DeepSeek client: {e}")
            return None
        
    # --- Update inside AIRephraser Class ---

    def diarize_transcript(self, transcript: str, agent_name: str, seller_name: str) -> str:
        """
        Uses DeepSeek to identify speakers and add labels (Diarization).
        """
        if not self.client or not transcript:
            return transcript

        # Fallbacks if names are missing
        agent_label = agent_name if agent_name and agent_name != "Not mentioned" else "Agent"
        seller_label = seller_name if seller_name and seller_name != "Not mentioned" else "Seller"

        system_prompt = f"""
        You are a professional transcription editor. 
        Your task is to format the following raw text into a dialogue script with speaker labels.
        
        The speakers are:
        1. **{agent_label}** (The Real Estate Agent/Investor). Context clues: Asks about selling, asking price, condition, offers, closing time.
        2. **{seller_label}** (The Property Owner). Context clues: Answers questions, talks about the house, negotiations, family situation.
        
        INSTRUCTIONS:
        - Add the label "{agent_label}:" or "{seller_label}:" at the start of each turn.
        - Fix minor grammar/punctuation errors but keep the wording authentic.
        - Break up long blocks of text into natural dialogue turns.
        - If there is a 3rd person (like a relative), label them "Relative".
        
        Raw Transcript:
        {transcript}
        """

        try:
            chat_completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Format and label this transcript."}
                ],
                temperature=0.1,
                max_tokens=2000  # Allow enough space for the full conversation
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception:
            return transcript # Return original if it fails

    def rephrase(self, topic_name: str, transcript: str) -> str:
        """Analyzes transcript using the DeepSeek API for a specific topic."""
        if not self.client:
            return f"DeepSeek API client not initialized. Cannot analyze {topic_name}."
            
        if not transcript or len(transcript) < 2:
            return "Transcript too short for analysis."

        question = ""
        system_prompt = ""

        if topic_name == "Reason for Selling":
            system_prompt = f"""
            You are an expert real estate call analyst.
            Your job is to analyze the following call transcript and rephrase
            the seller's stated reason for selling into a single, complete sentence.
            
            CRITICAL INSTRUCTIONS:
            - Write it as a narrative statement (e.g., "The seller is moving to...")
            - DO NOT use bullet points.
            - DO NOT add "The seller stated:". Just write the sentence.
            - EXAMPLE: "He is going to live with his father to take care of him because he has cancer."
            - If no reason is mentioned, say "No reason discussed in conversation".
            - Use the seller's own words as much as possible.
            - Be specific and concise.

            Transcript:
            {transcript}
            """
            question = "What is the seller's stated reason for selling the property?"

        elif topic_name == "Property Condition":
            system_prompt = f"""
            You are an expert real estate inspector writing a property condition report.
            Extract specific condition details from the transcript (renovations, age of roof/HVAC, repairs needed, damages, or "vacant").
            
            CRITICAL RULES:
            1. Return ONLY the facts. (e.g., "Brand new kitchen installed. Full bath in basement.")
            2. DO NOT use filler phrases like "The seller mentioned," "No other details," or "It was discussed that."
            3. DO NOT say what is missing. (e.g., Do NOT say "No roof details were given.")
            4. If the transcript contains NO useful condition information, return exactly: "None"
            
            Transcript:
            {transcript}
            """
            question = "Extract the property condition facts professionally."

        elif topic_name == "Mortgage Status":
            system_prompt = f"""
            You are a data extraction bot. Analyze the transcript and extract the mortgage status.
            
            CRITICAL OUTPUT RULES:
            1. Return ONLY the status (Maximum 5 words).
            2. DO NOT write full sentences like "The property is..."
            3. DO NOT repeat the address.
            4. Use these exact formats:
               - "Free and clear"
               - "Mortgage exists ($[Amount])"
               - "Mortgage exists (Amount unknown)"
               - "No mortgage information"

            Transcript:
            {transcript}
            """
            question = "What is the short mortgage status?"

        elif topic_name == "Occupancy Status":
            system_prompt = f"""
            You are an expert real estate call analyst.
            Analyze this conversation transcript and determine the occupancy status of the property.
            
            CRITICAL INSTRUCTIONS:
            - Be VERY specific about occupancy status
            - Options: "Owner Occupied", "Tenant Occupied", "Vacant", "Vacant Lot"
            - If tenant occupied with notice period, specify: "Tenant Occupied (30-day notice given)"
            - If no occupancy information is discussed, say "No occupancy information discussed"
            - Use clear, direct language
            -dont explain the reason just the occupancy status
            -make sure the answer is correct and short

            Transcript:
            {transcript}
            """
            question = "What is the occupancy status of this property based on the conversation?"

        elif topic_name == "Seller Personality":
            system_prompt = f"""
            You are an expert real estate call analyst.
            Analyze this conversation transcript to understand the seller's personality and communication style.
            
            CRITICAL INSTRUCTIONS:
            - Summarize their communication style (e.g., "Friendly and talkative," "Strictly business," "Seems stressed and in a hurry," "Calm and patient," "Sounds elderly and a bit confused").
            - Note any personal details they *voluntarily* shared that provide context (e.g., "Mentioned a new job," "Spoke about his father being sick," "Complained about tenants").
            - DO NOT make assumptions or state information that isn't in the transcript.
            - Combine this into a short, 1-2 sentence summary.
            - If no clear personality details are available, say "Seller was professional and did not share personal details."

            Transcript:
            {transcript}
            """
            question = "Summarize the seller's personality and communication style based on the transcript."
            
        elif topic_name == "Property Type":
            system_prompt = f"""
            You are an expert real estate data formatter. Your job is to clean and standardize a raw property type description from a form.

            CRITICAL FORMATTING RULES:
            1. Final Format: "PropertyType (X unit), Y Bedrooms, Z Bathrooms, SQFT Square Feet" OR "PropertyType, Y Bedrooms, Z Bathrooms, SQFT Square Feet" if single unit
            2. Commas are required: Use commas to separate all elements.
            3. No Slashes: DO NOT use slashes (/).
            4. No Abbreviations: DO NOT use abbreviations like 'bed', 'beds', 'ba', 'sf', 'sqft'. Always write out "Bedrooms", "Bathrooms", "Square Feet".
            5. Capitalization: Capitalize property types (e.g., "Single Family", "Duplex", "MultiFamily").
            6. Units: ONLY include unit count in parentheses if there are 2 or more units, e.g., "(2 unit)", "(3 unit)", "(4 unit)". If there is only 1 unit, DO NOT include "(1 unit)" - just write the property type without parentheses.
            7. Plurals: Use plurals correctly: "1 Bedroom", "2 Bedrooms", "1 Bathroom", "2.5 Bathrooms".
            8. Vacant Land: If it is land, the format is "Vacant land, X acres".

            You must always reformat the input. Never return the raw input.
            Always follow the format exactly.
            make sure everything is correct , the number of the bedrooms and bathrooms are correct
            please make sure of it before giving the final answer
            
            Property type from form to clean:
            {transcript}
            """
            question = "Clean and format this property type description from the form according to the rules."

        elif topic_name == "Important Highlights":
            system_prompt = f"""
            You are an expert real estate call analyst.
            Analyze this conversation transcript and identify the 3-5 most important highlights that a real estate investor should know.
            
            FOCUS ON:
            - **Financing constraints: Does the seller require CASH? Do they refuse Land Contracts/Creative finance?**
            - Urgent motivations or timelines
            - Financial pressures (taxes, mortgage, repairs needed)
            - Property condition issues or recent updates
            - Unique selling circumstances (inheritance, divorce, relocation)
            - Willingness to negotiate or flexibility
            
            FORMATTING RULES:
            - Return as bullet points (use • character)
            - Maximum 5 bullet points
            - Each bullet should be 1-2 sentences max
            - Be specific and actionable
            
            
            If no important highlights are found, return: "No critical highlights identified in the conversation."
            
            Transcript:
            {transcript}
            """
            question = "What are the most important highlights from this conversation that a real estate investor should know?"

        else:
            return f"No analysis defined for topic: {topic_name}"

        try:
            chat_completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            answer = chat_completion.choices[0].message.content.strip()
            
            if topic_name == "Property Type":
                if '/' in answer or 'beds' in answer.lower():
                    return transcript 
            
            return answer

        except Exception as e:
            st.error(f"❌ DEEPSEEK API ERROR: {e}")
            return f"Error analyzing {topic_name} with API."

# --- AIQualifier Class ---
# --- AIQualifier Class ---
class AIQualifier:
    def __init__(self, client):
        self.client = client
        self.model = "deepseek-chat"

    def _get_fallback_results(self, error_msg: str) -> Dict[str, Any]:
        return {
            'total_score': 0, 
            'verdict': "ERROR",
            'breakdown': {
                k: {'score': 0, 'notes': f"AI qualification failed: {error_msg}"} 
                for k in ['price', 'reason', 'closing', 'condition']
            }
        }

    def _get_val(self, data: Dict[str, FieldData], key: str) -> str:
        return data.get(key, FieldData("Not Provided", "")).value

    def qualify(self, lead_data: Dict[str, FieldData]) -> Dict[str, Any]:
        if not self.client: 
            return self._get_fallback_results("API client not initialized")

        # 1. Define data_summary FIRST (Safely)
        try:
            data_summary = f"""LEAD DATA:
            - Asking Price: {self._get_val(lead_data, 'asking_price')}
            - Zillow Estimate: {self._get_val(lead_data, 'zillow_estimate')}
            - Reason for Selling: {self._get_val(lead_data, 'reason_for_selling')}
            - Closing Time: {self._get_val(lead_data, 'closing_time')}
            - Property Condition: {self._get_val(lead_data, 'condition')}"""
        except Exception as e: 
            return self._get_fallback_results(f"Failed to format data: {e}")

        # 2. Define System Prompt (Using the defined data_summary)
        # Note: Added strict rules to stop "Thinking out loud"
        system_prompt = f"""
        You are an expert real estate lead qualification analyst. Analyze and score.
        
        RULES:
        1. Reason (50 pts): Solid reason (relocation, divorce, financial, age) = 50. Weak/None = 0.
        2. Price (20 pts): Asking price is LOWER than market estimates = 20. Higher = 0.
        3. Closing (20 pts): Timeline is 6 months or less = 20. > 6 months = 0.
        4. Condition (10 pts): Specific details (repairs/renovations) are present = 10. Vague/None = 0.
        
        LEAD DATA:
        {data_summary}
        
        CRITICAL OUTPUT RULES:
        - Return Valid JSON only.
        - The "notes" field must be the FINAL justification only. 
        - DO NOT include internal thinking, "corrections", or "rechecks".
        - If you award 0 points, explain why it failed directly.
        
        Return JSON format: 
        {{ 
            "total_score": <int>, 
            "verdict": "PRIME LEAD|Review|REJECT", 
            "breakdown": {{ 
                "price": {{ "score": <int>, "notes": "<string>" }}, 
                "reason": {{ "score": <int>, "notes": "<string>" }},
                "closing": {{ "score": <int>, "notes": "<string>" }},
                "condition": {{ "score": <int>, "notes": "<string>" }}
            }} 
        }}
        """

        try:
            with st.spinner("⚖️ Calling DeepSeek AI for final qualification..."):
                chat_completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt}, 
                        {"role": "user", "content": "Analyze and score."}
                    ],
                    temperature=0.0, 
                    max_tokens=500, 
                    response_format={"type": "json_object"}
                )
            
            results = json.loads(chat_completion.choices[0].message.content.strip())
            
            if 'total_score' not in results:
                raise ValueError("Missing total_score in AI response")
                
            st.success(f"⭐ AI Lead Score: {results['total_score']}/100 ({results['verdict']})")
            return results

        except json.JSONDecodeError as e:
            st.error(f"❌ AI QUALIFICATION ERROR: Failed to decode JSON: {e}")
            return self._get_fallback_results(f"AI returned invalid JSON: {e}")
        except Exception as e:
            st.error(f"❌ AI QUALIFICATION ERROR: {e}")
            return self._get_fallback_results(str(e))

# --- DataValidator Class ---
class DataValidator:
    """Validate and resolve data contradictions"""
    
    def clean_reason_field(self, reason_text: str, max_length: int = 500) -> str:
        if not reason_text or reason_text.strip() == "" or reason_text == "None":
            return "No reason discussed in conversation"
        
        if any(phrase in reason_text.lower() for phrase in ['no reason', 'did not discuss', 'not specified']):
            return reason_text
        
        if len(reason_text) > max_length:
            return reason_text[:max_length-3] + "..."
        
        return reason_text.strip()
    
    def validate_lead_data(self, form_data: Dict[str, FieldData]) -> Dict[str, Any]:
        """Validate lead data and return warnings/errors"""
        warnings = []
        errors = []
        confidence_scores = {}
        
        # Critical fields validation
        critical_fields = {
            'address': 'Address',
            'seller_name': 'Seller Name',
            'phone_number': 'Phone Number',
            'property_type': 'Property Type'
        }
        
        for field_key, field_name in critical_fields.items():
            field_data = form_data.get(field_key)
            if not field_data or not field_data.value or field_data.value.strip() == "" or field_data.value == "Not mentioned":
                errors.append(f"Missing critical field: {field_name}")
                confidence_scores[field_key] = 0.0
            else:
                confidence_scores[field_key] = field_data.confidence
        
        # Price validation
        asking_price = form_data.get('asking_price', FieldData("", "")).value
        if asking_price and asking_price != "Waiting for our offer":
            try:
                # Extract numeric value
                price_str = str(asking_price).replace('$', '').replace(',', '').replace(' ', '')
                # Remove non-numeric except decimal point
                price_str = ''.join(c for c in price_str if c.isdigit() or c == '.')
                if price_str:
                    price_value = float(price_str)
                    
                    # Flag suspicious prices
                    if price_value < 10000:
                        warnings.append(f"Suspiciously low asking price: ${price_value:,.0f}")
                    elif price_value > 10000000:
                        warnings.append(f"Very high asking price: ${price_value:,.0f} (verify accuracy)")
                    
                    # Compare with estimates
                    zillow_est = form_data.get('zillow_estimate', FieldData("", "")).value
                    if zillow_est and zillow_est != "Not mentioned":
                        try:
                            zillow_str = str(zillow_est).replace('$', '').replace(',', '').replace(' ', '')
                            zillow_str = ''.join(c for c in zillow_str if c.isdigit() or c == '.')
                            if zillow_str:
                                zillow_value = float(zillow_str)
                                diff_percent = abs((price_value - zillow_value) / zillow_value) * 100
                                if diff_percent > 50:
                                    warnings.append(f"Asking price differs significantly from Zillow estimate ({diff_percent:.1f}% difference)")
                        except:
                            pass
            except:
                warnings.append("Unable to validate asking price format")
        
        # Phone number validation
        phone = form_data.get('phone_number', FieldData("", "")).value
        if phone and phone != "Not mentioned":
            digits = ''.join(c for c in phone if c.isdigit())
            if len(digits) < 10:
                errors.append("Invalid phone number format")
                confidence_scores['phone_number'] = 0.3
            elif len(digits) == 10 or len(digits) == 11:
                confidence_scores['phone_number'] = 0.9
        
        # Confidence scoring
        overall_confidence = sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0.5
        
        return {
            'warnings': warnings,
            'errors': errors,
            'confidence_scores': confidence_scores,
            'overall_confidence': overall_confidence,
            'is_valid': len(errors) == 0
        }
    
    def display_validation_results(self, validation_results: Dict[str, Any]):
        """Display validation results with color-coded warnings"""
        if not validation_results:
            return
        
        errors = validation_results.get('errors', [])
        warnings = validation_results.get('warnings', [])
        confidence = validation_results.get('overall_confidence', 0.5)
        
        # Display errors
        if errors:
            st.markdown("### ⚠️ Data Validation Issues")
            for error in errors:
                st.markdown(f'<div class="validation-error">❌ {error}</div>', unsafe_allow_html=True)
        
        # Display warnings
        if warnings:
            if not errors:
                st.markdown("### ⚠️ Data Validation Warnings")
            for warning in warnings:
                st.markdown(f'<div class="validation-warning">⚠️ {warning}</div>', unsafe_allow_html=True)
        
        # Display confidence indicator
        if confidence >= 0.8:
            confidence_class = "confidence-high"
            confidence_text = "High"
        elif confidence >= 0.5:
            confidence_class = "confidence-medium"
            confidence_text = "Medium"
        else:
            confidence_class = "confidence-low"
            confidence_text = "Low"
        
        if errors or warnings:
            st.markdown(f'<div class="validation-warning">📊 <span class="{confidence_class}">Data Confidence: {confidence_text} ({confidence*100:.0f}%)</span></div>', unsafe_allow_html=True)
        elif confidence < 0.8:
            st.info(f"📊 Data Confidence: {confidence_text} ({confidence*100:.0f}%)")

# --- DataMerger Class ---
class DataMerger:
    """Intelligently merge form data with conversation insights"""
    
    def merge(self, form_data: Dict[str, FieldData], transcript: str, audio_analysis: Dict[str, Any]) -> Dict[str, FieldData]:
        merged = form_data.copy()
        return merged

# --- ReportGenerator Class ---
class ReportGenerator:
    """Generate professional text reports with enhanced data."""
    
    def __init__(self, ai_client=None):
        self.conversation_summarizer = ConversationSummarizer(ai_client=ai_client)
        
        self.form_fields = [
            ('list_name', 'List Name'),
            ('property_type', 'Property Type'),
            ('seller_name', 'Seller Name'),
            ('phone_number', 'Phone Number'),
            ('address', 'Address'),
            ('zillow_link', 'Zillow link'),
            ('asking_price', 'Asking Price'),
            ('zillow_estimate', 'Zillow Estimate'),
            ('realtor_estimate', 'Realtor Estimate'),
            ('redfin_estimate', 'Redfin Estimate'),
            ('reason_for_selling', 'Reason For Selling'),
            ('mortgage', 'Mortgage'),
            ('condition', 'Condition'),
            ('occupancy', 'Occupancy'),
            ('closing_time', 'Closing time'),
            ('best_time_to_call', 'Best time to call back'),
            ('agent_name', 'Agent Name'),
            ('call_recording', 'Call recording'),
        ]

        self.ai_analysis_fields = [
            ('personality', 'Seller Personality'),
            ('highlights', 'Important Call Highlights'),
        ]

        self.call_data_fields = [
            ('call_recording', 'Call recording')
        ]

    def _format_field_line(self, data: Dict[str, FieldData], field_key: str, display_name: str) -> str:
        if field_key in data:
            field_data = data[field_key]
            value = field_data.value

            if value and value != "Not mentioned" and value != "":
                if len(str(value)) > 400:
                    value = str(value)[:397] + "..."
                return f"◇{display_name}: {value}"
            else:
                return f"◇{display_name}: Not mentioned"
        else:
            return f"◇{display_name}: Not mentioned"
        
    # --- Add this method inside ReportGenerator Class ---
    def generate_clean_data_block(self, merged_data: Dict[str, FieldData]) -> str:
        """Generates a clean text block of just the updated data (Input format)."""
        lines = []
        for field_key, display_name in self.form_fields:
            # Get the final, merged, AI-enhanced value
            val = merged_data.get(field_key, FieldData("", "")).value
            
            # Clean up "Not mentioned" for raw data - leave blank or keep as is?
            # Usually for CRM import, we prefer the value or empty string.
            if not val or val == "Not mentioned":
                clean_val = ""
            else:
                clean_val = str(val).strip()
                
            lines.append(f"◇{display_name}: {clean_val}")
            
        return "\n".join(lines)

    def _format_section(self, title: str, fields_list: List[tuple], data: Dict[str, FieldData]) -> List[str]:
        lines = [
            "-" * 50,
            f"{title.upper()}",
            "-" * 50,
            ""
        ]
        for field_key, display_name in fields_list:
            lines.append(self._format_field_line(data, field_key, display_name))
        
        lines.append("")
        return lines

    def _format_qualification_section(self, results: Dict[str, Any]) -> List[str]:
        lines = [
            "=" * 50,
            f"LEAD QUALIFICATION: {results['verdict']} ({results['total_score']} / 100)",
            "=" * 50,
            ""
        ]
        
        def format_score(name, data_key):
            data = results['breakdown'].get(data_key, {'score': 0, 'notes': 'N/A'})
            return f"• {name}: {data['score']} pts"

        lines.append(format_score("Reason for Selling (50%)", "reason"))
        lines.append(format_score("Asking Price (20%)", "price"))
        lines.append(format_score("Closing Time (20%)", "closing"))
        lines.append(format_score("Property Condition (10%)", "condition"))
        
        lines.append("\nQUALIFICATION NOTES:")
        lines.append(f"- REASON: {results['breakdown'].get('reason', {}).get('notes', 'N/A')}")
        lines.append(f"- PRICE: {results['breakdown'].get('price', {}).get('notes', 'N/A')}")
        lines.append(f"- CLOSING: {results['breakdown'].get('closing', {}).get('notes', 'N/A')}")
        lines.append(f"- CONDITION: {results['breakdown'].get('condition', {}).get('notes', 'N/A')}")
        
        return lines + [""]

    def generate_report(self, merged_data: Dict[str, FieldData], 
                        transcript: Optional[str],
                        audio_result: Dict[str, Any],
                        nlp_data: Dict[str, str],
                        qualification_results: Dict[str, Any],
                        source_filename: str) -> str:
        
        lines = [
            "ENHANCED REAL ESTATE PROPERTY REPORT",
            "=" * 50,
            f"Source File: {os.path.basename(source_filename)}",
            ""
        ]

        lines.extend(self._format_section("PROPERTY & SELLER DETAILS", self.form_fields, merged_data))
        
        ai_data = {}
        if 'personality' in nlp_data:
            ai_data['personality'] = FieldData(
                value=nlp_data['personality'],
                source='conversation',
                confidence=0.9
            )
        
        if ai_data:
            lines.extend(self._format_section("AI CONVERSATION ANALYSIS", self.ai_analysis_fields, ai_data))
        
        lines.extend(self._format_qualification_section(qualification_results))

        # Add the new conversation analysis format
        if transcript:
            # Generate the structured analysis (Summary, Discussion Highlights, Action Items)
            conversation_analysis = self.conversation_summarizer.summarize(transcript, nlp_data)
            lines.extend([
                "",
                "=" * 50,
                "CALL ANALYSIS",
                "=" * 50,
                "",
                conversation_analysis,
                ""
            ])
            
            # Add Full Transcript section
            lines.extend(self._format_full_transcript(merged_data, transcript))
        else:
            lines.extend([
                "",
                "=" * 50,
                "CALL ANALYSIS",
                "=" * 50,
                "",
                "No call recording available for analysis.",
                "",
                "-" * 50,
                "FULL CALL TRANSCRIPT",
                "-" * 50,
                "",
                "No call recording available for analysis.",
                ""
            ])
        
        lines.extend(self._format_section("CALL & SOURCE DATA", self.call_data_fields, merged_data))
        
        return "\n".join(lines)
    
    def _format_full_transcript(self, data: Dict[str, FieldData], transcript: str) -> List[str]:
        lines = [
            "-" * 50,
            "FULL CALL TRANSCRIPT",
            "-" * 50,
            ""
        ]
        
        # DeepSeek has already formatted it, so we just add it line by line
        for line in transcript.splitlines():
            if line.strip():
                lines.append(line.strip())
                
        return lines + [""]
    
    def save_report(self, report_content: str, source_filename: str, 
                    output_dir: str = "output") -> str:
        base_name = os.path.splitext(os.path.basename(source_filename))[0]
        output_path = f"{base_name}_enhanced_report.txt"
        return output_path

# --- RealEstateAutomationSystem Class ---
class RealEstateAutomationSystem:
    def __init__(self):
        self.form_parser = FormParser()
        self.audio_processor = AudioProcessor()
        self.data_merger = DataMerger()
        self.data_validator = DataValidator()
        self.ai_qualifier = None

        self.nlp_analyzer = None
        self.rephraser = None
        self.report_generator = None  # Will be initialized after AI client is ready

        self._initialize_analyzers()

    def _initialize_analyzers(self):
        st.info("Loading AI models (this is cached and only runs once)...")
        
        self.nlp_analyzer = NLPAnalyzer()
        self.rephraser = AIRephraser()

        if self.rephraser.client:
            self.ai_qualifier = AIQualifier(client=self.rephraser.client)
            self.report_generator = ReportGenerator(ai_client=self.rephraser.client)
        else:
            st.error("❌ AI Qualifier NOT initialized (API client missing).")
            self.report_generator = ReportGenerator()  # Fallback without AI

    def process_lead(self, input_file_path: str, input_filename: str) -> tuple[str, str]:
        # --- Enhanced Progress Tracking ---
        progress_tracker = ProcessStatus()
        
        # Create progress display area outside status
        progress_placeholder = st.empty()
        
        with st.status("🔄 System Processing...", expanded=True) as status:
            # Update progress: File Upload
            progress_tracker.update_stage('file_upload', 'processing', 'Uploading and parsing file...', 10)
            with progress_placeholder.container():
                progress_tracker.display_enhanced_status()
            
            st.write("📂 Uploading and parsing file...")
            form_data = self.form_parser.parse_file(input_file_path)
            progress_tracker.update_stage('file_upload', 'complete', 'File parsed successfully', 100)
            progress_tracker.update_stage('data_parsing', 'processing', 'Extracting form data...', 20)
            with progress_placeholder.container():
                progress_tracker.display_enhanced_status()
            
            call_recording_url = form_data.get('call_recording').value if form_data.get('call_recording') else None
            
            audio_result = {'success': False}
            transcript = None
            nlp_analysis = {}
            
            if call_recording_url and call_recording_url.strip():
                progress_tracker.update_stage('data_parsing', 'complete', 'Data parsed', 100)
                progress_tracker.update_stage('audio_transcription', 'processing', 'Transcribing audio (Groq API)...', 10)
                with progress_placeholder.container():
                    progress_tracker.display_enhanced_status()
                
                st.write("🎧 Transcribing audio (Groq API)...")
                audio_result = self.audio_processor.transcribe_audio(call_recording_url)

                if not audio_result['success']:
                    progress_tracker.update_stage('audio_transcription', 'failed', 'Transcription failed')
                    status.update(label="❌ Processing Failed", state="error")
                    st.error("Transcription failed.")
                    return "Process stopped", "error.txt"
                
                transcript = audio_result['transcript']
                progress_tracker.update_stage('audio_transcription', 'complete', 'Transcription complete', 100)
                progress_tracker.update_stage('ai_analysis', 'processing', 'AI Analysis in progress...', 20)
                with progress_placeholder.container():
                    progress_tracker.display_enhanced_status()
                
                st.write("🗣️ Identifying speakers (Diarization)...")
                agent_name = form_data.get('agent_name').value
                seller_name = form_data.get('seller_name').value
                transcript = self.rephraser.diarize_transcript(transcript, agent_name, seller_name)
                audio_result['transcript'] = transcript
                progress_tracker.update_stage('ai_analysis', 'processing', 'Diarization complete, analyzing...', 40)
                with progress_placeholder.container():
                    progress_tracker.display_enhanced_status()

                st.write("🧠 Analyzing conversation psychology...")
                nlp_analysis = self.nlp_analyzer.analyze_transcript(transcript)
                progress_tracker.update_stage('ai_analysis', 'processing', 'Extracting insights...', 60)
                with progress_placeholder.container():
                    progress_tracker.display_enhanced_status()
                
                st.write("🤖 Extracting DeepSeek insights...")

                # AI Analysis Steps
                curr_pt = form_data.get('property_type').value or ""
                if curr_pt and "Not Specified" not in curr_pt:
                    ai_pt = self.rephraser.rephrase("Property Type", curr_pt)
                    if ai_pt and "Transcript too short" not in ai_pt:
                        form_data['property_type'] = FieldData(ai_pt, 'conversation', 0.95)
                
                nlp_analysis['reason'] = self.rephraser.rephrase("Reason for Selling", transcript)
                nlp_analysis['condition'] = self.rephraser.rephrase("Property Condition", transcript)
                nlp_analysis['mortgage'] = self.rephraser.rephrase("Mortgage Status", transcript)
                nlp_analysis['tenant'] = self.rephraser.rephrase("Occupancy Status", transcript)
                nlp_analysis['highlights'] = self.rephraser.rephrase("Important Highlights", transcript)
                nlp_analysis['personality'] = self.rephraser.rephrase("Seller Personality", transcript)
                progress_tracker.update_stage('ai_analysis', 'processing', 'Merging insights...', 80)
                with progress_placeholder.container():
                    progress_tracker.display_enhanced_status()

                st.write("🔄 Merging data sources...")
                form_data = self._apply_conversation_insights(form_data, nlp_analysis)
                progress_tracker.update_stage('ai_analysis', 'complete', 'AI Analysis complete', 100)
                with progress_placeholder.container():
                    progress_tracker.display_enhanced_status()
                
            else:
                st.warning("⚠️ No call recording URL found.")
                progress_tracker.update_stage('data_parsing', 'complete', 'Data parsed', 100)
                with progress_placeholder.container():
                    progress_tracker.display_enhanced_status()

            st.write("🔗 Finalizing derived fields...")
            form_data = self.data_merger.merge(form_data, transcript, audio_result)
            progress_tracker.update_stage('qualification', 'processing', 'Calculating lead score...', 50)
            with progress_placeholder.container():
                progress_tracker.display_enhanced_status()

            st.write("⚖️ Calculating Lead Score...")
            qualification_results = self.ai_qualifier.qualify(form_data)
            progress_tracker.update_stage('qualification', 'complete', 'Qualification complete', 100)
            progress_tracker.update_stage('report_generation', 'processing', 'Generating report...', 50)
            with progress_placeholder.container():
                progress_tracker.display_enhanced_status()
            
            st.write("📄 Generating report...")
            
            raw_data_content = self.report_generator.generate_clean_data_block(form_data)
            raw_data_filename = f"processed_{input_filename}"
            report_content = self.report_generator.generate_report(form_data, transcript, audio_result, nlp_analysis, qualification_results, input_filename)
            
            output_filename = self.report_generator.save_report(report_content, input_filename)
            progress_tracker.update_stage('report_generation', 'complete', 'Report generated', 100)
            with progress_placeholder.container():
                progress_tracker.display_enhanced_status()
            
            status.update(label="✅ Processing Complete!", state="complete", expanded=False)
        
        # Clear progress placeholder after completion
        progress_placeholder.empty()
        
        # --- NEW: Dashboard Results Layout ---
        st.divider()
        st.subheader("📊 Lead Dashboard")
        
        # --- MOVE DOWNLOADS TO SIDEBAR ---
        with st.sidebar:
            st.divider()
            st.header("💾 Exports")
            
            # 1. Data Only Download
            raw_data_content = self.report_generator.generate_clean_data_block(form_data)
            st.download_button(
                label="⬇️ Processed Data (.txt)",
                data=raw_data_content,
                file_name=f"data_{input_filename}",
                mime="text/plain"
            )
            
            # 2. Full Report Download
            st.download_button(
                label="⬇️ Full AI Report (.txt)",
                data=report_content,
                file_name=output_filename,
                mime="text/plain",
                type='primary'
            )
        # --- Data Validation Warnings ---
        validation_results = self.data_validator.validate_lead_data(form_data)
        if validation_results.get('warnings') or validation_results.get('errors'):
            self.data_validator.display_validation_results(validation_results)
            st.divider()
        
        # Top Row Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            score = qualification_results.get('total_score', 0)
            delta_color = "normal"
            if score > 75:
                delta_color = "normal"
            elif score < 50:
                delta_color = "inverse"
            st.metric("Lead Score", f"{score}/100", delta="High Priority" if score > 75 else ("Low Priority" if score < 50 else None), delta_color=delta_color)
        with col2:
            verdict = qualification_results.get('verdict', 'N/A')
            st.metric("Verdict", verdict)
        with col3:
            price = form_data.get('asking_price', FieldData("N/A","")).value
            st.metric("Asking Price", str(price)[:15])
        with col4:
            confidence = validation_results.get('overall_confidence', 0.5)
            confidence_pct = f"{confidence*100:.0f}%"
            st.metric("Data Confidence", confidence_pct)

        # Tabs Layout
       # --- IMPROVED TABS LAYOUT ---
        tab1, tab2, tab3, tab4 = st.tabs(["💾 DOWNLOADS", "🧠 AI INSIGHTS", "🎵 TRANSCRIPT & AUDIO", "📋 RAW DATA"])
        
        with tab1:
            st.subheader("📋 Final Processed Data")
            st.caption("💡 **Tip:** Click inside the box below, press **Ctrl + A** to select all, then **Ctrl + C** to copy.")
            
            # Generate the clean text block using the method we added earlier
            clean_text_block = self.report_generator.generate_clean_data_block(form_data)
            
            # Display it in a large, copy-paste friendly box
            st.text_area(
                label="Final Data",
                value=clean_text_block,
                height=600,
                label_visibility="collapsed" # Hides the small label above the box
            )
            
        with tab2:
            # --- Top Section: Highlights & Details ---
            c1, c2 = st.columns(2)
            with c1:
                st.info("**💡 Key Highlights**")
                highlights_raw = nlp_analysis.get('highlights', 'No highlights')
                formatted_highlights = highlights_raw.replace("$", "\$").replace("•", "\n\n-")
                st.markdown(formatted_highlights)
                
                st.divider()
                st.info("**🏠 Condition Notes**")
                st.write(form_data.get('condition', FieldData("N/A", "")).value)
                
            with c2:
                st.success("**💰 Financials**")
                st.write(f"**Mortgage:** {form_data.get('mortgage', FieldData('N/A', '')).value}")
                st.write(f"**Reason:** {form_data.get('reason_for_selling', FieldData('N/A', '')).value}")
                
                st.divider()
                st.warning("**👤 Seller Profile**")
                st.write(nlp_analysis.get('personality', 'N/A'))

            # --- NEW SECTION: Detailed Scoring Logic ---
            st.divider()
            st.subheader("💯 Qualification Scorecard & Logic")
            
            # Helper function to display a score row cleanly
           # Helper function to display a score row cleanly
            def display_score_row(label, key, max_points):
                item = qualification_results.get('breakdown', {}).get(key, {})
                score = item.get('score', 0)
                
                # FIX: Escape dollar signs so Streamlit doesn't turn them into weird math text
                raw_explanation = item.get('notes', 'No explanation provided.')
                explanation = raw_explanation.replace("$", "\$") 
                
                # Create a clean row
                sc1, sc2 = st.columns([1, 4])
                with sc1:
                    st.metric(label, f"{score} / {max_points}")
                with sc2:
                    if score > 0:
                        st.success(f"**Passed:** {explanation}")
                    else:
                        st.error(f"**Failed:** {explanation}")

            # Render the 4 rows
            display_score_row("Motivation", "reason", 50)
            display_score_row("Price", "price", 20)
            display_score_row("Timeline", "closing", 20)
            display_score_row("Condition", "condition", 10)

        with tab3: 
            # Audio Player Section
            call_recording_url = form_data.get('call_recording').value if form_data.get('call_recording') else None
            if call_recording_url and call_recording_url.strip():
                render_audio_player(call_recording_url, transcript)
                st.divider()
            else:
                st.info("ℹ️ No call recording URL available for playback")
                st.divider()
            
            # Transcript Section
            st.markdown("### 📝 Call Transcript (Diarized)")
            if transcript:
                st.text_area("Transcript", transcript, height=500, label_visibility="collapsed", key="transcript_display")
            else:
                st.warning("No transcript available")
            
        with tab4: 
            st.json(qualification_results)
            st.write(form_data)
            
        return report_content, output_filename
    
    def _apply_conversation_insights(self, form_data: Dict[str, FieldData], 
                                     nlp_analysis: Dict[str, str]) -> Dict[str, FieldData]:
        
        # ... (Keep 'reason' logic above this same) ...
        conversation_reason = nlp_analysis.get('reason', '')
        if conversation_reason and "no reason" not in conversation_reason.lower():
            cleaned_reason = self.data_validator.clean_reason_field(conversation_reason)
            form_data['reason_for_selling'] = FieldData(
                value=cleaned_reason, 
                source='conversation',
                confidence=1.0
            )

        # --- IMPROVED PROFESSIONAL CONDITION MERGE ---
        ai_condition = nlp_analysis.get('condition', '').strip()
        form_condition = form_data.get('condition', FieldData("", "")).value
        
        final_condition = form_condition # Start with what we have in the form

        # Only touch it if AI found something valid and it's not "None"
        if ai_condition and ai_condition != "None" and "no specific" not in ai_condition.lower():
            
            # Clean up the form condition to remove placeholders
            if not form_condition or form_condition == "Not mentioned":
                final_condition = ai_condition
            else:
                # INTELLIGENT MERGE: Combine them professionally
                # Avoid duplicating if the AI found the exact same text
                if ai_condition.lower() not in form_condition.lower():
                    # Ensure ends with punctuation
                    if final_condition and final_condition[-1] not in ['.', '!', '?']:
                        final_condition += "."
                    
                    final_condition = f"{final_condition} {ai_condition}"
        
        form_data['condition'] = FieldData(
            value=final_condition,
            source='merged',
            confidence=1.0
        )
        # ---------------------------------------------

        # ... (Keep the rest of the logic for mortgage, occupancy, etc. the same) ...
        conversation_mortgage = nlp_analysis.get('mortgage', '')
        if conversation_mortgage and "no mortgage information" not in conversation_mortgage.lower():
            form_data['mortgage'] = FieldData(value=conversation_mortgage, source='conversation', confidence=0.95)
            
        conversation_occupancy = nlp_analysis.get('tenant', '')
        if conversation_occupancy and "no occupancy information" not in conversation_occupancy.lower():
            form_data['occupancy'] = FieldData(value=conversation_occupancy, source='conversation', confidence=0.95)
            
        conversation_motivation = nlp_analysis.get('motivation', '')
        if conversation_motivation and "no motivation" not in conversation_motivation.lower():
            form_data['motivation_details'] = FieldData(value=conversation_motivation, source='conversation', confidence=0.9)
            
        conversation_highlights = nlp_analysis.get('highlights', '')
        if conversation_highlights and "no critical highlights" not in conversation_highlights.lower():
            nlp_analysis['highlights'] = conversation_highlights
        
        return form_data

# --- MAIN STREAMLIT UI ---
st.set_page_config(page_title="AI Real Estate Lead Manager", page_icon="🏠", layout="wide")
local_css() # Apply the CSS

# --- Sidebar: Configuration & Inputs ---
with st.sidebar:
    st.title("🎛️ Controls")
    
    st.subheader("🔑 System Status")
    if os.getenv("DEEPSEEK_API_KEY") or st.secrets.get("DEEPSEEK_API_KEY"):
        st.success("🧠 Intelligence: Online")
    else:
        st.error("🧠 Intelligence: Offline")
        
    if os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY"):
        st.success("👂 Hearing: Online")
    else:
        st.error("👂 Hearing: Offline")
    
    st.divider()
    
    st.subheader("📥 Input Method")
    input_method = st.radio("Select Source:", ["📝 Paste Text", "📁 Upload File"])
    
    lead_data = None
    source_name = "direct_input"

    if input_method == "📝 Paste Text":
        lead_data = st.text_area("Paste Form Data:", height=200, placeholder="◇List Name: ...")
        if lead_data: source_name = "pasted_data"
    else:
        uploaded_file = st.file_uploader("Upload .txt lead", type=["txt"])
        if uploaded_file:
            lead_data = uploaded_file.getvalue().decode('utf-8')
            source_name = uploaded_file.name

    st.divider()
    process_btn = st.button("🚀 Process Lead", type="primary", use_container_width=True)

# --- Main Area ---
st.markdown("""
<div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 30px;'>
    <h1 style='color: white; margin: 0;'>🏠 AI Real Estate Manager</h1>
    <p style='color: white; font-size: 18px; margin: 10px 0 0 0;'>Automated Lead Qualification, Transcription & Analysis System</p>
</div>
""", unsafe_allow_html=True)

if not lead_data:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 30px; background-color: #f0f2f6; border-radius: 10px; margin: 20px 0;'>
            <h3>👈 Get Started</h3>
            <p>Please provide lead data in the sidebar to begin processing.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced example with more details
    with st.expander("📖 See Example Input Format", expanded=False):
        st.markdown("**Example Lead Data Format:**")
        st.code("""◇List Name: out of state
◇Property Type: Single Family
◇Seller Name: John Doe
◇Phone Number: (555) 123-4567
◇Address: 123 Main St, City, State 12345
◇Zillow link: https://www.zillow.com/...
◇Asking Price: $250,000
◇Zillow Estimate: $240,000
◇Realtor Estimate: $245,000
◇Redfin Estimate: $242,000
◇Reason For Selling: Relocation
◇Mortgage: Free and clear
◇Condition: Good condition, recently renovated
◇Occupancy: Owner occupied
◇Closing time: As soon as possible
◇Best time to call back: Anytime
◇Agent Name: David White
◇Call recording: https://example.com/audio.mp3""", language="text")
        
        st.info("💡 **Tip:** You can either paste the data directly or upload a `.txt` file with this format.")

if lead_data and process_btn:
    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as tmp_file:
        tmp_file.write(lead_data)
        temp_file_path = tmp_file.name

    try:
        system = RealEstateAutomationSystem()
        system.process_lead(temp_file_path, source_name)
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.exception(e)
    finally:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
