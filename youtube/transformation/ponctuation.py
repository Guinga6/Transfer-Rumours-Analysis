import os
import json
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle
import time
from collections import Counter

class SentenceBoundaryDetector:
    def __init__(self, silence_threshold=0.3, context_window=7):
        """
        Initialize the sentence boundary detector
        
        Args:
            silence_threshold: Minimum silence duration (in seconds) to consider as boundary candidate
            context_window: Number of words to extract before and after each boundary
        """
        self.silence_threshold = silence_threshold
        self.context_window = context_window
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Define lexical patterns for sentence boundaries (Football/Fabrizio Romano specific)
        self.head_patterns = [
            # Fabrizio's signature phrases
            'here we go', 'breaking', 'exclusive', 'confirmed', 'done deal',
            'medical scheduled', 'personal terms', 'total agreement',
            # Common sentence starters in football reporting
            'so', 'now', 'then', 'well', 'but', 'and', 'however', 'meanwhile',
            'also', 'in addition', 'furthermore', 'moreover', 'therefore',
            # Transfer-specific starters
            'the player', 'the club', 'sources', 'according to', 'as reported',
            'understand', 'told', 'informed', 'revealed', 'expected',
            # Time/sequence indicators
            'today', 'tomorrow', 'yesterday', 'this morning', 'this evening',
            'next week', 'soon', 'imminent', 'final', 'last'
        ]
        
        self.tail_patterns = [
            # Fabrizio's signature endings
            'here we go', 'done deal', 'confirmed', 'agreed', 'completed',
            'sealed', 'official', 'announced', 'signed', 'finalized',
            # Football-specific endings
            'million euros', 'million pounds', 'per season', 'plus bonuses',
            'add ons', 'release clause', 'buy back', 'loan deal',
            'permanent deal', 'free transfer', 'contract extension',
            # Certainty indicators
            'definitely', 'certainly', 'absolutely', 'guaranteed',
            'no doubt', 'for sure', 'confirmed', 'verified',
            # Common conversational endings
            'you know', 'right', 'okay', 'exactly', 'obviously',
            'clearly', 'basically', 'essentially', 'really', 'actually',
            # Transfer status endings
            'pending', 'expected', 'likely', 'possible', 'rumored',
            'reported', 'claimed', 'suggested', 'indicated'
        ]
        
        # Common sentence-ending punctuation for noisy labels
        self.sentence_endings = ['.', '!', '?', ';']

    def print_progress(self, current, total, prefix='Progress'):
        """Print progress bar"""
        percent = f"{100 * (current / float(total)):.1f}"
        filled_length = int(50 * current // total)
        bar = '█' * filled_length + '-' * (50 - filled_length)
        print(f'\r{prefix} |{bar}| {percent}%', end='\r')
        if current == total:
            print()

    def load_json_data(self, json_folder):
        """Load all JSON files with audio analysis and transcription data"""
        data_files = []
        json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
        
        print(f"Chargement de {len(json_files)} fichiers JSON...")
        
        for i, file in enumerate(json_files):
            self.print_progress(i + 1, len(json_files), "Chargement")
            
            file_path = os.path.join(json_folder, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check if required fields exist
                if 'transcription' in data and 'audio_analysis' in data:
                    if data['audio_analysis'].get('silence_segments'):
                        data_files.append({
                            'filename': file,
                            'transcription': data['transcription'],
                            'silence_segments': data['audio_analysis']['silence_segments'],
                            'speech_segments': data['audio_analysis']['speech_segments']
                        })
            except Exception as e:
                print(f"\nErreur lors du chargement de {file}: {e}")
        
        print(f"✓ {len(data_files)} fichiers chargés avec succès")
        return data_files

    def tokenize_text(self, text):
        """Simple tokenization of text"""
        # Remove extra whitespace and convert to lowercase
        text = re.sub(r'\s+', ' ', text.lower().strip())
        # Split on word boundaries
        words = re.findall(r'\b\w+\b', text)
        return words

    def find_boundary_candidates(self, silence_segments):
        """Find silence segments that could be sentence boundaries"""
        candidates = []
        for segment in silence_segments:
            silence_duration = segment[1] - segment[0]
            if silence_duration >= self.silence_threshold:
                candidates.append({
                    'time': segment[1],  # End of silence (start of next speech)
                    'duration': silence_duration
                })
        return candidates

    def extract_lexical_features(self, words, word_index):
        """Extract lexical features around a word boundary"""
        features = {}
        
        # Get context window
        start_idx = max(0, word_index - self.context_window)
        end_idx = min(len(words), word_index + self.context_window + 1)
        
        prev_words = words[start_idx:word_index] if word_index > 0 else []
        next_words = words[word_index:end_idx] if word_index < len(words) else []
        
        # Convert to text for pattern matching
        prev_text = ' '.join(prev_words).lower()
        next_text = ' '.join(next_words).lower()
        
        # Head pattern features (beginning of potential sentence)
        features['has_head_pattern'] = 0
        features['head_pattern_strength'] = 0
        if next_words:
            first_words = ' '.join(next_words[:4]).lower()  # Extended for "here we go"
            for pattern in self.head_patterns:
                if first_words.startswith(pattern) or pattern in first_words:
                    features['has_head_pattern'] = 1
                    # Stronger weight for Fabrizio's signature phrases
                    if pattern in ['here we go', 'breaking', 'exclusive', 'done deal']:
                        features['head_pattern_strength'] = 2
                    else:
                        features['head_pattern_strength'] = 1
                    break
        
        # Tail pattern features (end of potential sentence)
        features['has_tail_pattern'] = 0
        features['tail_pattern_strength'] = 0
        if prev_words:
            last_words = ' '.join(prev_words[-4:]).lower()  # Extended for "million euros"
            for pattern in self.tail_patterns:
                if pattern in last_words:
                    features['has_tail_pattern'] = 1
                    # Stronger weight for transfer-specific endings
                    if pattern in ['here we go', 'done deal', 'million euros', 'million pounds']:
                        features['tail_pattern_strength'] = 2
                    else:
                        features['tail_pattern_strength'] = 1
                    break
        
        # Football-specific features
        features['has_transfer_keywords'] = 0
        transfer_keywords = [
            'transfer', 'deal', 'contract', 'agreement', 'signing', 'medical',
            'personal terms', 'fee', 'clause', 'loan', 'permanent', 'extension'
        ]
        context_text = (prev_text + ' ' + next_text).lower()
        for keyword in transfer_keywords:
            if keyword in context_text:
                features['has_transfer_keywords'] = 1
                break
        
        # Number/money indicators (common in transfer news)
        features['has_numbers'] = 0
        if re.search(r'\d+', context_text):
            features['has_numbers'] = 1
        
        features['has_money_terms'] = 0
        money_terms = ['million', 'euros', 'pounds', 'fee', 'cost', 'price', 'value']
        for term in money_terms:
            if term in context_text:
                features['has_money_terms'] = 1
                break
        
        # Player/club name indicators (usually capitalized)
        features['has_proper_nouns'] = 0
        if next_words and any(word[0].isupper() for word in next_words[:3] if word):
            features['has_proper_nouns'] = 1
        
        # Word length features
        features['prev_avg_word_length'] = np.mean([len(w) for w in prev_words]) if prev_words else 0
        features['next_avg_word_length'] = np.mean([len(w) for w in next_words]) if next_words else 0
        
        # Context size features
        features['prev_word_count'] = len(prev_words)
        features['next_word_count'] = len(next_words)
        
        # Capitalization feature (stronger indicator in football reporting)
        features['next_word_capitalized'] = 0
        if word_index < len(words) and words[word_index]:
            features['next_word_capitalized'] = 1 if words[word_index][0].isupper() else 0
        
        return features

    def extract_prosodic_features(self, silence_duration, speech_segments, boundary_time):
        """Extract prosodic features from audio analysis"""
        features = {}
        
        # Silence duration
        features['silence_duration'] = silence_duration
        features['silence_duration_log'] = np.log(silence_duration + 0.001)
        
        # Find speech segments before and after boundary
        prev_segment = None
        next_segment = None
        
        for segment in speech_segments:
            if segment[1] <= boundary_time:
                prev_segment = segment
            elif segment[0] >= boundary_time and next_segment is None:
                next_segment = segment
                break
        
        # Previous speech segment duration
        if prev_segment:
            features['prev_speech_duration'] = prev_segment[1] - prev_segment[0]
        else:
            features['prev_speech_duration'] = 0
        
        # Next speech segment duration
        if next_segment:
            features['next_speech_duration'] = next_segment[1] - next_segment[0]
        else:
            features['next_speech_duration'] = 0
        
        # Relative silence duration
        total_speech_around = features['prev_speech_duration'] + features['next_speech_duration']
        if total_speech_around > 0:
            features['silence_speech_ratio'] = silence_duration / total_speech_around
        else:
            features['silence_speech_ratio'] = 0
        
        return features

    def create_noisy_labels(self, transcription):
        """Create noisy labels based on punctuation"""
        # Find positions of sentence-ending punctuation
        sentence_boundaries = []
        for i, char in enumerate(transcription):
            if char in self.sentence_endings:
                sentence_boundaries.append(i)
        return sentence_boundaries

    def align_boundaries_with_words(self, words, transcription, boundary_candidates):
        """Align silence boundaries with word positions"""
        # This is a simplified alignment - in practice you'd need more sophisticated
        # time-to-word alignment using forced alignment tools
        
        aligned_boundaries = []
        total_chars = len(transcription)
        total_words = len(words)
        
        for candidate in boundary_candidates:
            # Estimate word position based on time ratio
            # This is rough - real implementation would use proper alignment
            estimated_char_pos = int((candidate['time'] / 10.0) * total_chars)  # Assuming 10s total duration
            estimated_word_pos = int((estimated_char_pos / total_chars) * total_words)
            
            # Clamp to valid range
            estimated_word_pos = max(0, min(total_words - 1, estimated_word_pos))
            
            aligned_boundaries.append({
                'word_index': estimated_word_pos,
                'time': candidate['time'],
                'duration': candidate['duration']
            })
        
        return aligned_boundaries

    def extract_features_from_file(self, file_data):
        """Extract all features from a single file"""
        transcription = file_data['transcription']
        words = self.tokenize_text(transcription)
        
        if len(words) == 0:
            return []
        
        # Find boundary candidates
        boundary_candidates = self.find_boundary_candidates(file_data['silence_segments'])
        
        if not boundary_candidates:
            return []
        
        # Align boundaries with words
        aligned_boundaries = self.align_boundaries_with_words(
            words, transcription, boundary_candidates
        )
        
        # Create noisy labels
        punctuation_boundaries = self.create_noisy_labels(transcription)
        
        features_list = []
        
        for boundary in aligned_boundaries:
            word_idx = boundary['word_index']
            
            # Extract lexical features
            lexical_features = self.extract_lexical_features(words, word_idx)
            
            # Extract prosodic features
            prosodic_features = self.extract_prosodic_features(
                boundary['duration'],
                file_data['speech_segments'],
                boundary['time']
            )
            
            # Combine features
            combined_features = {**lexical_features, **prosodic_features}
            
            # Create noisy label (1 if near punctuation, 0 otherwise)
            # This is simplified - you'd want better alignment
            estimated_char_pos = int((word_idx / len(words)) * len(transcription))
            is_sentence_boundary = any(
                abs(estimated_char_pos - p_pos) < 10 for p_pos in punctuation_boundaries
            )
            
            combined_features['label'] = 1 if is_sentence_boundary else 0
            combined_features['filename'] = file_data['filename']
            combined_features['word_index'] = word_idx
            
            features_list.append(combined_features)
        
        return features_list

    def prepare_training_data(self, json_folder):
        """Prepare training data from JSON files"""
        print("Préparation des données d'entraînement...")
        
        # Load data files
        data_files = self.load_json_data(json_folder)
        
        if not data_files:
            raise ValueError("Aucun fichier de données trouvé!")
        
        # Extract features from all files
        all_features = []
        
        print("Extraction des caractéristiques...")
        for i, file_data in enumerate(data_files):
            self.print_progress(i + 1, len(data_files), "Extraction")
            
            try:
                features = self.extract_features_from_file(file_data)
                all_features.extend(features)
            except Exception as e:
                print(f"\nErreur lors du traitement de {file_data['filename']}: {e}")
        
        if not all_features:
            raise ValueError("Aucune caractéristique extraite!")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_features)
        
        # Separate features and labels
        feature_columns = [col for col in df.columns if col not in ['label', 'filename', 'word_index']]
        X = df[feature_columns].fillna(0)
        y = df['label']
        
        print(f"\n✓ {len(all_features)} échantillons extraits")
        print(f"✓ {len(feature_columns)} caractéristiques")
        print(f"✓ Distribution des labels: {Counter(y)}")
        
        return X, y, feature_columns

    def train(self, json_folder):
        """Train the sentence boundary detection model"""
        print("=== ENTRAÎNEMENT DU MODÈLE ===")
        
        # Prepare training data
        X, y, feature_columns = self.prepare_training_data(json_folder)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        print("Normalisation des caractéristiques...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train classifier
        print("Entraînement du classificateur...")
        start_time = time.time()
        self.classifier.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        
        # Evaluate
        print("Évaluation du modèle...")
        train_score = self.classifier.score(X_train_scaled, y_train)
        test_score = self.classifier.score(X_test_scaled, y_test)
        
        y_pred = self.classifier.predict(X_test_scaled)
        
        # Print results
        print(f"\n=== RÉSULTATS ===")
        print(f"Temps d'entraînement: {training_time:.2f}s")
        print(f"Score d'entraînement: {train_score:.3f}")
        print(f"Score de test: {test_score:.3f}")
        print(f"\nRapport de classification:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 caractéristiques importantes:")
        print(feature_importance.head(10))
        
        self.is_trained = True
        self.feature_columns = feature_columns
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'feature_importance': feature_importance
        }

    def save_model(self, model_path):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Le modèle n'a pas encore été entraîné!")
        
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'silence_threshold': self.silence_threshold,
            'context_window': self.context_window
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Modèle sauvegardé: {model_path}")

    def load_model(self, model_path):
        """Load a trained model"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.silence_threshold = model_data['silence_threshold']
        self.context_window = model_data['context_window']
        self.is_trained = True
        
        print(f"✓ Modèle chargé: {model_path}")

    def predict_boundaries(self, transcription, silence_segments, speech_segments):
        """Predict sentence boundaries for new text"""
        if not self.is_trained:
            raise ValueError("Le modèle n'a pas encore été entraîné!")
        
        words = self.tokenize_text(transcription)
        if len(words) == 0:
            return []
        
        # Find boundary candidates
        boundary_candidates = self.find_boundary_candidates(silence_segments)
        if not boundary_candidates:
            return []
        
        # Create file data structure
        file_data = {
            'transcription': transcription,
            'silence_segments': silence_segments,
            'speech_segments': speech_segments,
            'filename': 'prediction'
        }
        
        # Extract features
        features_list = self.extract_features_from_file(file_data)
        if not features_list:
            return []
        
        # Prepare features for prediction
        df = pd.DataFrame(features_list)
        X = df[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.classifier.predict(X_scaled)
        probabilities = self.classifier.predict_proba(X_scaled)[:, 1]
        
        # Return results
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            if pred == 1:  # Sentence boundary predicted
                results.append({
                    'word_index': features_list[i]['word_index'],
                    'probability': prob,
                    'word': words[features_list[i]['word_index']] if features_list[i]['word_index'] < len(words) else ""
                })
        
        return results

    def insert_punctuation(self, transcription, boundaries, punctuation_rules=None):
        """
        Insert punctuation based on predicted boundaries
        
        Args:
            transcription: Original transcription text
            boundaries: List of boundary predictions from predict_boundaries()
            punctuation_rules: Dict with rules for punctuation selection (optional)
        
        Returns:
            Punctuated transcription
        """
        if not boundaries:
            return transcription
        
        # Extract words while preserving original case and positions
        words_with_positions = []
        for match in re.finditer(r'\b\w+\b', transcription):
            words_with_positions.append({
                'word': match.group(),
                'start': match.start(),
                'end': match.end()
            })
        
        if not words_with_positions:
            return transcription
        
        # Default punctuation rules
        if punctuation_rules is None:
            punctuation_rules = {
                'default': '.',
                'high_confidence': '.',  # prob > 0.8
                'medium_confidence': '.',  # Changed from comma to period
                'low_confidence': ',',   # prob > 0.3
                'question_words': '?',
                'exclamation_words': '!'
            }
        
        # Sort boundaries by word index
        sorted_boundaries = sorted(boundaries, key=lambda x: x['word_index'])
        
        # Question and exclamation indicators
        question_starters = ['what', 'when', 'where', 'why', 'how', 'who', 'which', 'can', 'could', 'would', 'should', 'will', 'do', 'does', 'did', 'is', 'are', 'was', 'were']
        exclamation_words = ['wow', 'amazing', 'incredible', 'fantastic', 'unbelievable', 'breaking', 'confirmed', 'done', 'here we go', 'exclusive']
        
        # Create result by inserting punctuation
        result = transcription
        offset = 0  # Track position changes due to insertions
        
        for boundary in sorted_boundaries:
            word_idx = boundary['word_index']
            prob = boundary['probability']
            
            # Skip if word index is out of range
            if word_idx >= len(words_with_positions):
                continue
            
            # Get the word position
            word_info = words_with_positions[word_idx]
            insert_pos = word_info['end'] + offset
            
            # Analyze context for punctuation selection
            words = [w['word'] for w in words_with_positions]
            
            # Look at next few words for question detection
            next_words = words[word_idx+1:word_idx+4] if word_idx+1 < len(words) else []
            next_text = ' '.join(next_words).lower()
            
            # Look at current and previous words for exclamation detection
            context_start = max(0, word_idx-2)
            context_end = min(len(words), word_idx+2)
            current_context = ' '.join(words[context_start:context_end]).lower()
            
            # Determine punctuation
            punct = punctuation_rules['default']
            
            if any(word.lower() in next_text for word in question_starters):
                punct = punctuation_rules.get('question_words', '?')
            elif any(word in current_context for word in exclamation_words):
                punct = punctuation_rules.get('exclamation_words', '!')
            elif prob >= 0.8:
                punct = punctuation_rules.get('high_confidence', '.')
            elif prob >= 0.6:
                punct = punctuation_rules.get('medium_confidence', '.')
            elif prob >= 0.3:
                punct = punctuation_rules.get('low_confidence', ',')
            else:
                punct = punctuation_rules['default']
            
            # Insert punctuation
            result = result[:insert_pos] + punct + result[insert_pos:]
            offset += len(punct)
        
        return result

    def process_and_punctuate_files(self, json_folder, output_folder=None, min_probability=0.5):
        """
        Process all JSON files, apply punctuation, and save results
        
        Args:
            json_folder: Input folder with JSON files
            output_folder: Output folder (if None, overwrites original files)
            min_probability: Minimum probability threshold for boundaries
        """
        if not self.is_trained:
            raise ValueError("Le modèle n'a pas encore été entraîné!")
        
        if output_folder and not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Load data files
        data_files = self.load_json_data(json_folder)
        
        print(f"\n=== PONCTUATION DES TRANSCRIPTIONS ===")
        print(f"Traitement de {len(data_files)} fichiers...")
        
        processed_count = 0
        
        for i, file_data in enumerate(data_files):
            self.print_progress(i + 1, len(data_files), "Ponctuation")
            
            try:
                # Load original file
                original_file_path = os.path.join(json_folder, file_data['filename'])
                with open(original_file_path, 'r', encoding='utf-8') as f:
                    original_data = json.load(f)
                
                # Predict boundaries
                boundaries = self.predict_boundaries(
                    file_data['transcription'],
                    file_data['silence_segments'],
                    file_data['speech_segments']
                )
                
                # Filter boundaries by probability threshold
                filtered_boundaries = [b for b in boundaries if b['probability'] >= min_probability]
                
                # Apply punctuation
                punctuated_transcription = self.insert_punctuation(
                    file_data['transcription'], 
                    filtered_boundaries
                )
                
                # Add punctuated transcription to original data
                original_data['transcription_ponctuer'] = punctuated_transcription
                original_data['sentence_boundaries'] = {
                    'boundaries': filtered_boundaries,
                    'total_boundaries': len(boundaries),
                    'filtered_boundaries': len(filtered_boundaries),
                    'min_probability': min_probability
                }
                
                # Save file
                if output_folder:
                    output_file_path = os.path.join(output_folder, file_data['filename'])
                else:
                    output_file_path = original_file_path
                
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(original_data, f, ensure_ascii=False, indent=2)
                
                processed_count += 1
                
            except Exception as e:
                print(f"\nErreur lors du traitement de {file_data['filename']}: {e}")
        
        print(f"\n✓ {processed_count} fichiers traités avec succès")
        if output_folder:
            print(f"✓ Résultats sauvegardés dans: {output_folder}")
        else:
            print(f"✓ Fichiers originaux mis à jour")


def main():
    # Configuration
    json_folder = "video_data/mistral"  # Dossier contenant les fichiers JSON
    model_path = "sentence_boundary_model.pkl"
    output_folder = "video_data/result"  # Optional: leave None to overwrite original files
    
    # Initialiser le détecteur
    detector = SentenceBoundaryDetector(
        silence_threshold=0.3,  # 300ms minimum de silence
        context_window=7        # 7 mots de contexte de chaque côté
    )
    
    try:
        # Entraîner le modèle
        print("Début de l'entraînement...")
        results = detector.train(json_folder)
        
        # Sauvegarder le modèle
        detector.save_model(model_path)
        
        # Traiter tous les fichiers et ajouter la ponctuation
        detector.process_and_punctuate_files(
            json_folder=json_folder,
            output_folder=output_folder,
            min_probability=0.5  # Seuil minimum de probabilité
        )
        
        # Exemple d'utilisation pour prédiction individuelle
        print("\n=== EXEMPLE DE PRÉDICTION ===")
        
        # Charger un fichier exemple pour tester
        test_file = os.path.join(json_folder, os.listdir(json_folder)[0])
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        if 'transcription' in test_data and 'audio_analysis' in test_data:
            boundaries = detector.predict_boundaries(
                test_data['transcription'],
                test_data['audio_analysis']['silence_segments'],
                test_data['audio_analysis']['speech_segments']
            )
            
            # Filter boundaries with reasonable probability
            filtered_boundaries = [b for b in boundaries if b['probability'] >= 0.3]
            
            punctuated = detector.insert_punctuation(
                test_data['transcription'],
                filtered_boundaries
            )
            
            print(f"Transcription originale:")
            print(f"  {test_data['transcription'][:300]}...")
            print(f"\nTranscription ponctuée:")
            print(f"  {punctuated[:300]}...")
            print(f"\nStatistiques:")
            print(f"  - Frontières détectées: {len(boundaries)}")
            print(f"  - Frontières retenues (prob >= 0.3): {len(filtered_boundaries)}")
            print(f"\nDétail des frontières retenues:")
            for boundary in filtered_boundaries[:8]:  # Afficher les 8 premières
                print(f"  - Mot {boundary['word_index']}: '{boundary['word']}' (prob: {boundary['probability']:.3f})")
            
            # Show word-by-word comparison if boundaries were found
            if filtered_boundaries:
                print(f"\nComparaison mot par mot (premiers 50 mots):")
                original_words = re.findall(r'\b\w+\b', test_data['transcription'])
                boundary_indices = set(b['word_index'] for b in filtered_boundaries)
                
                comparison_text = ""
                for i, word in enumerate(original_words[:50]):
                    if i in boundary_indices:
                        comparison_text += f"[{word}*] "
                    else:
                        comparison_text += f"{word} "
                print(f"  (* = frontière de phrase détectée)")
                print(f"  {comparison_text}")
    
    except Exception as e:
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()