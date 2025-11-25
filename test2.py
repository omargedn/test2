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

# --- Dataclass ---
@dataclass
class FieldData:
    value: Any
    source: str
    confidence: float = 1.0

class ProcessStatus:
    def __init__(self):
        self.stages = {
            'file_upload': {'status': 'waiting', 'message': 'File Upload'},
            'data_parsing': {'status': 'waiting', 'message': 'Data Parsing'}, 
            'audio_transcription': {'status': 'waiting', 'message': 'Audio Transcription'},
            'ai_analysis': {'status': 'waiting', 'message': 'AI Analysis'},
            'qualification': {'status': 'waiting', 'message': 'Lead Qualification'},
            'report_generation': {'status': 'waiting', 'message': 'Report Generation'}
        }
    
    def update_stage(self, stage, status, message=None):
        self.stages[stage]['status'] = status
        if message:
            self.stages[stage]['message'] = message
    
    def display_status(self):
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
            'moving_time': ['Moving time'],
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
            'moving_time': self._clean_closing_time,
            'motivation_details': self._clean_motivation,
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
        condition_lower = condition.lower()
        if 'total renovation' in condition_lower:
            return "Property requires complete renovation"
        if 'needs some repairs' in condition_lower:
            return "Property requires some repairs"
        if any(word in condition_lower for word in ['excellent', 'great', 'brand new']):
            return "Property in excellent condition"
        if any(word in condition_lower for word in ['good', 'nice', 'well maintained']):
            return "Property in good condition"
        if 'vacant lot' in condition_lower:
            return "Vacant lot"
        return condition
    
    def _clean_reason(self, reason: str) -> str:
        reason_lower = reason.lower()
        if any(phrase in reason_lower for phrase in ['fix', 'investment', 'flip']):
            return "Property investment business"
        if "taxes" in reason_lower:
            return "Financial pressure from property taxes"
        if any(word in reason_lower for word in ['relocat', 'move']):
            return "Relocation"
        return "Standard property disposition"
    
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
    """Generates a summary of the conversation."""
    
    def __init__(self):
        pass

    def summarize(self, transcript: str, nlp_data: Dict[str, str]) -> str:
        """Generate enhanced summary based on extracted NLP data"""
        if not transcript:
            return "No transcript available for summarization."
        
        key_points = []
        
        reason = nlp_data.get('reason', '')
        if reason and "no reason" not in reason.lower():
            key_points.append(f"Reason for Selling: {reason}")
        
        motivation = nlp_data.get('motivation', '')
        if motivation and "no motivation" not in motivation.lower():
            key_points.append(f"Seller Motivation: {motivation}")
            
        personality = nlp_data.get('personality', '')
        if personality and "did not share" not in personality.lower():
            key_points.append(f"Seller Personality: {personality}")

        condition = nlp_data.get('condition', '')
        if condition and "no specific" not in condition.lower():
            key_points.append(f"Property Condition: {condition}")
            
        mortgage = nlp_data.get('mortgage', '')
        if mortgage and "no mortgage" not in mortgage.lower():
            key_points.append(f"Mortgage Status: {mortgage}")

        occupancy = nlp_data.get('tenant', '')
        if occupancy and "no occupancy" not in occupancy.lower():
            key_points.append(f"Occupancy: {occupancy}")
        
        
        summary_lines = [
            "ENHANCED CONVERSATION ANALYSIS",
            "=" * 60,
            "",
            "KEY DISCUSSION POINTS:",
            "-" * 40
        ]
        
        if key_points:
            for point in key_points:
                summary_lines.append(f"• {point}")
        else:
            summary_lines.append("• Key details were not clearly discussed in the conversation.")
            
        summary_lines.append("")
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
            You are an expert real estate call analyst.
            Your job is to analyze the following call transcript and summarize
            the seller's description of the property's condition into a clear, concise statement.
            
            CRITICAL INSTRUCTIONS:
            - Focus on ACTUAL condition details mentioned: roof, HVAC, foundation, repairs needed, updates, age, etc.
            - If NO specific condition details are mentioned, return: "No specific condition details discussed"
            - Be realistic and factual - only mention what was actually discussed
            - Keep it to 1-2 sentences maximum
            - DO NOT make up or assume condition details
            - DO NOT use bullet points
            -Make the answer have all informaation about the condition of the property , but also not too long

            Transcript:
            {transcript}
            """
            question = "What specific details about the property's condition were mentioned in the conversation?"

        elif topic_name == "Mortgage Status":
            system_prompt = f"""
            You are an expert real estate call analyst.
            Analyze this conversation transcript and determine the mortgage status of the property.
            
            CRITICAL INSTRUCTIONS:
            - Be VERY specific about mortgage status
            - put in consider the input value of mortage of the form data , compare it with the transcript and give your final answer 
            - If mortgage exists, mention any amounts discussed
            - If owned free and clear, state that clearly
            - If no mortgage information is discussed, say "No mortgage information discussed"
            - Use clear, direct language
            -take in consideration that the whisper may have some errors , so be careful while giving the final answer
            -answer must be correct and short 

            Transcript:
            {transcript}
            """
            question = "What is the mortgage status of the property based on the conversation?"

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
            1. Final Format: "PropertyType (X unit), Y Bedrooms, Z Bathrooms, SQFT Square Feet"
            2. Commas are required: Use commas to separate all elements.
            3. No Slashes: DO NOT use slashes (/).
            4. No Abbreviations: DO NOT use abbreviations like 'bed', 'beds', 'ba', 'sf', 'sqft'. Always write out "Bedrooms", "Bathrooms", "Square Feet".
            5. Capitalization: Capitalize property types (e.g., "Single Family", "Duplex", "MultiFamily").
            6. Units: Always include a unit count in parentheses, e.g., "(1 unit)", "(2 unit)".
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
            - Urgent motivations or timelines
            - Financial pressures (taxes, mortgage, repairs needed)
            - Property condition issues or recent updates
            - Unique selling circumstances (inheritance, divorce, relocation)
            - Willingness to negotiate or flexibility
            - Any red flags or opportunities
            
            FORMATTING RULES:
            - Return as bullet points (use • character)
            - Maximum 5 bullet points
            - Each bullet should be 1-2 sentences max
            - Be specific and actionable
            - Focus on what matters for investment decisions
            
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
class AIQualifier:
    """Analyzes final lead data against a set of business rules"""
    
    def __init__(self, client):
        self.client = client
        self.model = "deepseek-chat"
        self.re = re

    def _get_fallback_results(self, error_msg: str) -> Dict[str, Any]:
        return {
            'total_score': 0,
            'verdict': "ERROR",
            'breakdown': {
                'price': {'score': 0, 'notes': f"AI qualification failed: {error_msg}"},
                'reason': {'score': 0, 'notes': f"AI qualification failed: {error_msg}"},
                'closing': {'score': 0, 'notes': f"AI qualification failed: {error_msg}"},
                'condition': {'score': 0, 'notes': f"AI qualification failed: {error_msg}"},
            }
        }

    def _get_val(self, data: Dict[str, FieldData], key: str) -> str:
        return data.get(key, FieldData("Not Provided", "")).value

    def qualify(self, lead_data: Dict[str, FieldData]) -> Dict[str, Any]:
        if not self.client:
            return self._get_fallback_results("API client not initialized")

        try:
            data_summary = f"""
            LEAD DATA:
            - Asking Price: {self._get_val(lead_data, 'asking_price')}
            - Zillow Estimate: {self._get_val(lead_data, 'zillow_estimate')}
            - Realtor Estimate: {self._get_val(lead_data, 'realtor_estimate')}
            - Redfin Estimate: {self._get_val(lead_data, 'redfin_estimate')}
            - Reason for Selling: {self._get_val(lead_data, 'reason_for_selling')}
            - Closing Time: {self._get_val(lead_data, 'closing_time')}
            - Property Condition: {self._get_val(lead_data, 'condition')}
            """
        except Exception as e:
            return self._get_fallback_results(f"Failed to format data: {e}")

        system_prompt = f"""
        You are an expert real estate lead qualification analyst. Your job is to analyze the following lead data and score it according to a strict set of rules.

        QUALIFICATION RULES:
        1.  **Reason for Selling (50 points):**
            -   The reason MUST be a "solid reason" (e.g., relocation, divorce, financial trouble, inheritance, major life event).
            -   "Weak reasons" (e.g., "don't need it anymore," "standard disposition," "no reason discussed," "not specified") get 0 points.
            -   **Award 50 points for a solid reason, 0 for a weak one.**

        2.  **Asking Price (20 points):**
            -   First, calculate the average of all available market estimates (Zillow, Realtor, Redfin).
            -   Then, check if the "Asking Price" is *below* that average market value.
            -   If no asking price or no market estimates are provided, this fails.
            -   **Award 20 points if it's below market, 0 otherwise.**

        3.  **Closing Time (20 points):**
            -   The "Closing Time" must be 6 months or less (e.g., "ASAP," "30 days," "flexible," "6 months").
            -   If the time is over 6 months (e.g., "7 months," "next year") or not provided, it fails.
            -   **Award 20 points if it's <= 6 months, 0 otherwise.**

        4.  **Property Condition (10 points):**
            -   Any specific details about the condition must be provided.
            -   If the condition is "not specified," "no details discussed," or "no transcript," it fails.
            -   **Award 10 points if *any* condition details are present, 0 otherwise.**

        TASK:
        Analyze this data blob, calculate the score for each rule, and provide a total score and verdict.

        LEAD DATA TO ANALYZE:
        {data_summary}

        FINAL INSTRUCTIONS:
        -   You MUST return your answer in a valid JSON format.
        -   The JSON MUST match this exact structure:
        {{
          "total_score": <number>,
          "verdict": "<string: 'PRIME LEAD' (>=80 pts), 'Review' (50-79 pts), or 'REJECT' (<50 pts)>",
          "breakdown": {{
            "price": {{ "score": <number>, "notes": "<string: Your brief justification>" }},
            "reason": {{ "score": <number>, "notes": "<string: Your brief justification>" }},
            "closing": {{ "score": <number>, "notes": "<string: Your brief justification>" }},
            "condition": {{ "score": <number>, "notes": "<string: Your brief justification>" }}
          }}
        }}
        -   Be strict with the rules.
        -   Do not include any text outside the JSON.
        """

        try:
            with st.spinner("⚖️ Calling DeepSeek AI for final qualification..."):
                chat_completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": "Analyze the lead data and return the qualification JSON."}
                    ],
                    temperature=0.0,
                    max_tokens=500,
                    response_format={"type": "json_object"}
                )
            
            response_text = chat_completion.choices[0].message.content.strip()
            results = json.loads(response_text)
            
            if 'total_score' not in results or 'breakdown' not in results:
                raise ValueError("AI response missing required keys")
                
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

# --- DataMerger Class ---
class DataMerger:
    """Intelligently merge form data with conversation insights"""
    
    def merge(self, form_data: Dict[str, FieldData], transcript: str, audio_analysis: Dict[str, Any]) -> Dict[str, FieldData]:
        merged = form_data.copy()
        
        moving_time_data = merged.get('moving_time')
        moving_time_value = moving_time_data.value if moving_time_data else ""

        if not moving_time_value or moving_time_value == "Not mentioned":
            property_type_data = merged.get('property_type')
            property_type = property_type_data.value if property_type_data else ""
            
            if 'vacant lot' in str(property_type).lower():
                merged['moving_time'] = FieldData(
                    value="Not applicable - vacant lot",
                    source='derived',
                    confidence=0.9
                )
            else:
                closing_time_data = merged.get('closing_time')
                closing_time = closing_time_data.value if closing_time_data else ""
                
                if closing_time and closing_time != "Not mentioned" and closing_time != "":
                    merged['moving_time'] = FieldData(
                        value=closing_time,
                        source='derived',
                        confidence=0.7
                    )
        
        return merged

# --- ReportGenerator Class ---
class ReportGenerator:
    """Generate professional text reports with enhanced data."""
    
    def __init__(self):
        self.conversation_summarizer = ConversationSummarizer()
        
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
            ('moving_time', 'Moving time'),
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

        if transcript:
            lines.extend(self._format_full_transcript(merged_data, transcript))
        else:
            lines.extend([
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
            "(Note: Speaker labels are a 'best guess' and only added where confident.)",
            ""
        ]

        agent_name_full = data.get('agent_name', FieldData("Agent", "", 0.0)).value
        seller_name_full = data.get('seller_name', FieldData("Seller", "", 0.0)).value

        agent_label = agent_name_full.split()[0].strip(":") if agent_name_full else "Agent"
        seller_label = seller_name_full.split()[0].strip(":") if seller_name_full else "Seller"

        max_label_len = max(len(agent_label), len(seller_label)) + 1

        for text in transcript.splitlines():
            text = text.strip()
            if not text:
                continue

            text_lower = text.lower()
            label = ""

            if (("this is " + agent_label.lower()) in text_lower or \
                ("my name is " + agent_label.lower()) in text_lower):
                label = agent_label

            elif "speaking" in text_lower and len(text_lower) < 20:
                label = seller_label

            elif (("this is " + seller_label.lower()) in text_lower or \
                ("my name is " + seller_label.lower()) in text_lower):
                label = seller_label

            if label:
                formatted_label = (label + ":").ljust(max_label_len)
                lines.append(f"{formatted_label} {text}")
            else:
                formatted_label = " ".ljust(max_label_len + 1)
                lines.append(f"{formatted_label} {text}")

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
        self.report_generator = ReportGenerator() 
        self.data_validator = DataValidator()
        self.ai_qualifier = None

        self.nlp_analyzer = None
        self.rephraser = None

        self._initialize_analyzers()

    def _initialize_analyzers(self):
        st.info("Loading AI models (this is cached and only runs once)...")
        
        self.nlp_analyzer = NLPAnalyzer()
        self.rephraser = AIRephraser()

        if self.rephraser.client:
            self.ai_qualifier = AIQualifier(client=self.rephraser.client)
        else:
            st.error("❌ AI Qualifier NOT initialized (API client missing).")

    def process_lead(self, input_file_path: str, input_filename: str) -> tuple[str, str]:
        """Process a single lead file and return report content and filename"""
        
        status_tracker = ProcessStatus()
        status_tracker.update_stage('file_upload', 'complete')
        
        st.subheader("🔄 Processing Status")
        status_tracker.display_status()
        
        st.info("📝 Parsing form data...")
        form_data = self.form_parser.parse_file(input_file_path)
        
        call_recording_url = form_data.get('call_recording').value if form_data.get('call_recording') else None
        
        status_tracker.update_stage('data_parsing', 'complete')
        status_tracker.display_status()

        audio_result = {'success': False}
        transcript = None
        nlp_analysis = {}
        
        if call_recording_url and call_recording_url.strip():
            st.info("🎵 Processing call recording...")
            audio_result = self.audio_processor.transcribe_audio(call_recording_url, status_tracker)

            if not audio_result['success']:
                st.error("🚫 PROCESS STOPPED: Transcription failed. Please check the audio URL and try again.")
                status_tracker.display_status()
                return "Process stopped due to transcription failure", "error.txt"
            
            if audio_result['success']:
                transcript = audio_result['transcript']
                
                with st.spinner("🤖 Analyzing conversation with fast NLP..."):
                    nlp_analysis = self.nlp_analyzer.analyze_transcript(transcript)

                st.info("🧠 STARTING DEEPSEEK AI ANALYSIS...")
                
                with st.spinner("🧠 Cleaning 'Property Type' with AI..."):
                    current_property_type = form_data.get('property_type').value if form_data.get('property_type') else ""
                    if current_property_type and current_property_type != "Property Type Not Specified":
                        ai_property_type = self.rephraser.rephrase("Property Type", current_property_type)
                        if ai_property_type and "Transcript too short" not in ai_property_type:
                            form_data['property_type'] = FieldData(
                                value=ai_property_type,
                                source='conversation', 
                                confidence=0.95
                            )
                    
                with st.spinner("🧠 Analyzing 'Reason for Selling' with AI..."):
                    ai_reason = self.rephraser.rephrase("Reason for Selling", transcript)
                    nlp_analysis['reason'] = ai_reason
                
                with st.spinner("🧠 Analyzing 'Property Condition' with AI..."):
                    ai_condition = self.rephraser.rephrase("Property Condition", transcript)
                    nlp_analysis['condition'] = ai_condition
                
                with st.spinner("🧠 Analyzing 'Mortgage Status' with AI..."):
                    ai_mortgage = self.rephraser.rephrase("Mortgage Status", transcript)
                    nlp_analysis['mortgage'] = ai_mortgage
                
                with st.spinner("🧠 Analyzing 'Occupancy Status' with AI..."):
                    ai_occupancy = self.rephraser.rephrase("Occupancy Status", transcript)
                    nlp_analysis['tenant'] = ai_occupancy

                with st.spinner("🧠 Identifying important call highlights..."):
                    ai_highlights = self.rephraser.rephrase("Important Highlights", transcript)
                    nlp_analysis['highlights'] = ai_highlights

                with st.spinner("🧠 Analyzing 'Seller Personality' with AI..."):
                    ai_personality = self.rephraser.rephrase("Seller Personality", transcript)
                    nlp_analysis['personality'] = ai_personality
                
                st.success("✅ DEEPSEEK AI ANALYSIS COMPLETE")

                status_tracker.update_stage('ai_analysis', 'complete')
                status_tracker.display_status()
                
                with st.spinner("🔄 Applying AI conversation insights to form data..."):
                    form_data = self._apply_conversation_insights(form_data, nlp_analysis)
                
            else:
                st.error(f"❌ Audio processing failed: {audio_result.get('error')}")
        else:
            st.warning("⚠️ No call recording URL found in form data. Skipping audio analysis.")
            nlp_analysis = {
                'reason': "No transcript available",
                'condition': "No transcript available", 
                'mortgage': "No transcript available",
                'tenant': "No transcript available",
                'motivation': "No transcript available",
                'personality': "No transcript available"
            }

        with st.spinner("🔗 Deriving dependent fields (Moving Time)..."):
            form_data = self.data_merger.merge(form_data, transcript, audio_result)

        st.info("⚖️ Starting final AI-powered lead qualification...")
        qualification_results = self.ai_qualifier.qualify(form_data)
        status_tracker.update_stage('qualification', 'complete')
        status_tracker.display_status()
        
        st.info("📊 Generating final report...")
        report_content = self.report_generator.generate_report(
            form_data, 
            transcript, 
            audio_result,
            nlp_analysis,
            qualification_results,
            input_filename
        )
        status_tracker.update_stage('report_generation', 'complete')
        status_tracker.display_status()
        
        output_filename = self.report_generator.save_report(report_content, input_filename)
        
        st.subheader("🎉 PROCESSING RESULTS")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 FORM DATA", 
            "🎵 TRANSCRIPT", 
            "🤖 AI ANALYSIS", 
            "⚖️ QUALIFICATION", 
            "📄 FINAL REPORT"
        ])
        
        with tab1:
            st.header("📋 Parsed Form Data")
            form_display = []
            for field_key, field_data in form_data.items():
                if field_data.value and field_data.value not in ["", "Not mentioned"]:
                    display_name = next((names[0] for key, names in self.form_parser.field_patterns.items() if key == field_key), field_key)
                    form_display.append(f"◇{display_name}: {field_data.value}")
            
            st.text_area("Form Data", "\n".join(form_display), height=500, key="form_data_tab")
        
        with tab2:
            st.header("🎵 Call Transcript")
            if transcript and transcript != "No transcription available":
                st.text_area("Transcript", transcript, height=500, key="transcript_tab")
            else:
                st.info("No transcript available for this lead")
        
        with tab3:
            st.header("🤖 AI Conversation Analysis")
            
            ai_analysis_content = []
            
            if 'reason' in nlp_analysis:
                ai_analysis_content.append(f"◇Reason for Selling: {nlp_analysis['reason']}")
            if 'condition' in nlp_analysis:
                ai_analysis_content.append(f"◇Property Condition: {nlp_analysis['condition']}")
            if 'mortgage' in nlp_analysis:
                ai_analysis_content.append(f"◇Mortgage Status: {nlp_analysis['mortgage']}")
            if 'tenant' in nlp_analysis:
                ai_analysis_content.append(f"◇Occupancy Status: {nlp_analysis['tenant']}")
            if 'personality' in nlp_analysis:
                ai_analysis_content.append(f"◇Seller Personality: {nlp_analysis['personality']}")
            if 'motivation' in nlp_analysis:
                ai_analysis_content.append(f"◇Motivation Analysis: {nlp_analysis['motivation']}")
            if 'highlights' in nlp_analysis:  
                ai_analysis_content.append(f"◇Important Call Highlights:\n{nlp_analysis['highlights']}")  
            
            if ai_analysis_content:
                st.text_area("AI Analysis Results", "\n".join(ai_analysis_content), height=500, key="ai_analysis_tab")
            else:
                st.info("No AI analysis available (no transcript)")
        
        with tab4:
            st.header("⚖️ Lead Qualification")
            
            qual_content = [
                f"◇Total Score: {qualification_results['total_score']}/100",
                f"◇Verdict: {qualification_results['verdict']}",
                "",
                "BREAKDOWN:"
            ]
            
            for category, data in qualification_results['breakdown'].items():
                qual_content.append(f"◇{category.title()}: {data['score']} pts - {data['notes']}")
            
            st.text_area("Qualification Results", "\n".join(qual_content), height=500, key="qualification_tab")
        
        with tab5:
            st.header("📄 Final Comprehensive Report")
            st.text_area("Complete Report", report_content, height=500, key="final_report_tab")
        
        st.success(f"✅ Enhanced report generated!")
        
        st.download_button(
            label="⬇️ Download Complete Report",
            data=report_content,
            file_name=output_filename,
            mime="text/plain"
        )
        
        return report_content, output_filename
    
    def _apply_conversation_insights(self, form_data: Dict[str, FieldData], 
                                     nlp_analysis: Dict[str, str]) -> Dict[str, FieldData]:
        conversation_reason = nlp_analysis.get('reason', '')
        if conversation_reason and "no reason" not in conversation_reason.lower():
            cleaned_reason = self.data_validator.clean_reason_field(conversation_reason)
            form_data['reason_for_selling'] = FieldData(
                value=cleaned_reason, 
                source='conversation',
                confidence=1.0
            )
        
        conversation_condition = nlp_analysis.get('condition', '')
        if conversation_condition and "no specific" not in conversation_condition.lower():
            form_data['condition'] = FieldData(
                value=conversation_condition,
                source='conversation',
                confidence=1.0
            )
        
        conversation_mortgage = nlp_analysis.get('mortgage', '')
        if conversation_mortgage and "no mortgage information" not in conversation_mortgage.lower():
            form_data['mortgage'] = FieldData(
                value=conversation_mortgage, 
                source='conversation',
                confidence=0.95
            )
        
        conversation_occupancy = nlp_analysis.get('tenant', '')
        if conversation_occupancy and "no occupancy information" not in conversation_occupancy.lower():
            form_data['occupancy'] = FieldData(
                value=conversation_occupancy, 
                source='conversation',
                confidence=0.95
            )
        
        conversation_motivation = nlp_analysis.get('motivation', '')
        if conversation_motivation and "no motivation" not in conversation_motivation.lower():
            form_data['motivation_details'] = FieldData(
                value=conversation_motivation,
                source='conversation',
                confidence=0.9
            )
        
        conversation_highlights = nlp_analysis.get('highlights', '')
        if conversation_highlights and "no critical highlights" not in conversation_highlights.lower():
            nlp_analysis['highlights'] = conversation_highlights
        
        return form_data

# --- STREAMLIT UI ---
st.set_page_config(page_title="Real Estate Lead Automation", layout="wide")
st.title("🏠 Real Estate Lead Automation System")
st.markdown("Automated lead processing with AI-powered analysis")

# API Key Loading
deepseek_api_key = None
try:
    deepseek_api_key = getattr(st, "secrets", {}).get("DEEPSEEK_API_KEY")
except Exception:
    deepseek_api_key = None 
if not deepseek_api_key:
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

if deepseek_api_key:
    st.success("✅ DeepSeek API key loaded")
    os.environ["DEEPSEEK_API_KEY"] = deepseek_api_key 
else:
    st.error("❌ DEEPSEEK_API_KEY not found in Streamlit secrets or environment variables.")
    st.warning("Please add your DEEPSEEK_API_KEY to your secrets to run the app.")
    st.stop() 

# Input Method Selection
input_method = st.radio(
    "Choose input method:",
    ["📝 Paste Lead Data", "📁 Upload File"],
    horizontal=True
)

lead_data = None
source_name = "direct_input"

if input_method == "📝 Paste Lead Data":
    st.subheader("📝 Paste Lead Form Data")
    lead_text = st.text_area(
        "Paste your lead form data here:",
        height=300,
        placeholder="Paste your lead form data in this format:\n◇List Name:-\n◇Property Type:-\n◇Seller Name:-\n◇Phone Number:-\n◇Address:-\n◇Zillow link:-\n◇Asking Price:-\n◇Zillow Estimate:-\n◇Realtor Estimate:-\n◇Redfin Estimate:-\n◇Reason For Selling:-\n◇Motivation details:-\n◇Mortgage:-\n◇Condition:-\n◇Occupancy:-\n◇Closing time:-\n◇Moving time:-\n◇Best time to call back:-\n◇Agent Name:-\n◇Call recording:-",
        label_visibility="collapsed"
    )
    
    if lead_text.strip():
        lead_data = lead_text
        source_name = "pasted_data"

else:
    st.subheader("📁 Upload Lead File")
    uploaded_file = st.file_uploader("Select a `.txt` lead file", type=["txt"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        lead_data = uploaded_file.getvalue().decode('utf-8')
        source_name = uploaded_file.name

# Process Button
if lead_data:
    st.markdown("---")
    
    with st.expander("📋 Data Preview", expanded=True):
        st.text(lead_data[:1000] + "..." if len(lead_data) > 1000 else lead_data)
    
    if st.button("🚀 Process Lead", type="primary"):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as tmp_file:
            tmp_file.write(lead_data)
            temp_file_path = tmp_file.name

        try:
            system = RealEstateAutomationSystem()
            report_content, report_filename = system.process_lead(temp_file_path, source_name)
            st.success("✅ Lead processing completed! Check the tabs above for detailed results.")

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.exception(e)
            
        finally:
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)