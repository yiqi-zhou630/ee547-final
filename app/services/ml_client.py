"""
ML Client Service
Loads and manages the trained ML model for scoring student answers
"""

import os
import sys
import platform
from pathlib import Path
from typing import Tuple
import logging

# Set environment variables BEFORE any torch/transformers imports
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TRANSFORMERS_NO_TORCH'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# macOS specific: Force CPU to avoid Bus errors with MPS - MUST BE BEFORE torch import
if platform.system() == 'Darwin':
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch

# Disable MPS IMMEDIATELY after torch import - CRITICAL for macOS
if platform.system() == 'Darwin':
    try:
        torch.backends.mps.is_available = lambda: False
        torch.backends.mps.is_built = lambda: False
    except AttributeError:
        pass
    try:
        torch.mps.is_available = lambda: False
        torch.mps.is_built = lambda: False
    except AttributeError:
        pass
    torch.set_default_device('cpu')

from app.core.config import settings

logger = logging.getLogger(__name__)

# Global model cache
_model_instance = None


def _get_model_path() -> Path:
    """Get absolute path to the ML model directory."""
    model_path = Path(settings.ML_MODEL_PATH)
    if not model_path.is_absolute():
        # Make it relative to project root
        project_root = Path(__file__).parent.parent.parent
        model_path = project_root / model_path
    return model_path


def reload_model():
    """
    Reload the ML model from disk.
    This forces a fresh load of the model weights.
    """
    global _model_instance
    
    try:
        model_path = _get_model_path()
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        logger.info(f"Loading ML model from {model_path}")
        
        # Import here to avoid import errors if model is not available
        from model_training.inference import ScoringModel
        
        # Create new instance - will use CPU by default (we set it above)
        _model_instance = ScoringModel(
            model_path=str(model_path),
            device='cpu',  # Force CPU
            use_features=False
        )
        
        logger.info("ML model loaded successfully")
        return _model_instance
        
    except Exception as e:
        logger.error(f"Failed to load ML model: {e}")
        raise


def _get_model():
    """
    Get or load the ML model instance.
    Uses a global cache to avoid reloading the model for every prediction.
    """
    global _model_instance
    
    if _model_instance is None:
        _model_instance = reload_model()
    
    return _model_instance


def _map_class_to_label(class_name: str) -> str:
    """
    Map 5-class model output to 3-class labels for API consistency.
    
    Model outputs 5 classes:
    - correct: answer is correct
    - contradictory: answer contradicts reference
    - partially_correct_incomplete: answer is partially correct but incomplete
    - irrelevant: answer is irrelevant to question
    - non_domain: answer is outside domain knowledge
    
    API labels (3-class simplified):
    - correct: maps from 'correct'
    - partial: maps from 'partially_correct_incomplete'
    - incorrect: maps from 'contradictory', 'irrelevant', 'non_domain'
    """
    mapping = {
        'correct': 'correct',
        'partially_correct_incomplete': 'partial',
        'contradictory': 'incorrect',
        'irrelevant': 'incorrect',
        'non_domain': 'incorrect',
    }
    
    return mapping.get(class_name, 'incorrect')


def score_answer(
    question_text: str,
    ref_answer: str,
    student_answer: str
) -> Tuple[str, float, float, str]:
    """
    Score a student's answer using the ML model.
    
    Args:
        question_text: The question being answered
        ref_answer: The reference/correct answer
        student_answer: The student's submitted answer
    
    Returns:
        Tuple of (label, score, confidence, explanation)
        - label: str, one of ['correct', 'partial', 'incorrect']
        - score: float, 0-5
        - confidence: float, 0-1 (confidence in the prediction)
        - explanation: str, human-readable explanation of the score
    
    Raises:
        Exception: If model loading or inference fails
    """
    try:
        # Get model instance (will load if not cached)
        model = _get_model()
        
        # Run model prediction
        result = model.predict(
            question=question_text,
            reference_answer=ref_answer,
            student_answer=student_answer
        )
        
        # Extract results
        score = result['score']  # 0-5
        raw_class = result['class']  # 5-class label from model
        class_probs = result['class_probabilities']  # Dict of all class probs
        
        # Map to 3-class label for API
        label = _map_class_to_label(raw_class)
        
        # Calculate confidence as max probability among mapped classes
        if label == 'correct':
            confidence = float(class_probs.get('correct', 0.0))
        elif label == 'partial':
            confidence = float(class_probs.get('partially_correct_incomplete', 0.0))
        else:  # incorrect
            confidence = max(
                float(class_probs.get('contradictory', 0.0)),
                float(class_probs.get('irrelevant', 0.0)),
                float(class_probs.get('non_domain', 0.0))
            )
        
        # Generate explanation based on score and label
        explanation = _generate_explanation(
            label=label,
            score=score,
            raw_class=raw_class,
            question=question_text,
            ref_answer=ref_answer,
            student_answer=student_answer
        )
        
        logger.info(
            f"Scored submission: label={label}, score={score}, "
            f"confidence={confidence:.3f}, raw_class={raw_class}"
        )
        
        return label, score, confidence, explanation
        
    except Exception as e:
        logger.error(f"Error scoring answer: {e}", exc_info=True)
        raise


def _generate_explanation(
    label: str,
    score: float,
    raw_class: str,
    question: str,
    ref_answer: str,
    student_answer: str
) -> str:
    """
    Generate a human-readable explanation for the score.
    
    Args:
        label: One of ['correct', 'partial', 'incorrect']
        score: Numerical score 0-5
        raw_class: Raw 5-class label from model
        question: The question text
        ref_answer: The reference answer
        student_answer: The student's answer
    
    Returns:
        Human-readable explanation string
    """
    if label == 'correct':
        return f"Your answer is correct. Score: {score}/5.0"
    
    elif label == 'partial':
        return (
            f"Your answer is partially correct but incomplete. "
            f"You correctly identified some key concepts, but missed some details. "
            f"Score: {score}/5.0"
        )
    
    else:  # incorrect
        if raw_class == 'contradictory':
            return (
                f"Your answer contradicts the reference answer. "
                f"Please review the correct approach. Score: {score}/5.0"
            )
        elif raw_class == 'irrelevant':
            return (
                f"Your answer does not address the question. "
                f"Please provide an answer directly related to the question. "
                f"Score: {score}/5.0"
            )
        elif raw_class == 'non_domain':
            return (
                f"Your answer is outside the domain of this subject. "
                f"Please stick to the relevant domain knowledge. Score: {score}/5.0"
            )
        else:
            return f"Your answer needs improvement. Score: {score}/5.0"
