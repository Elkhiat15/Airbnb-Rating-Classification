import numpy as np
import pandas as pd
from collections import Counter
import logging

try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from imblearn.combine import SMOTETomek, SMOTEENN
    from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("imbalanced-learn not installed. Run: pip install imbalanced-learn")
    print("Class balancing features will not be available.")

logger = logging.getLogger(__name__)


def apply_smote(X_train, y_train, strategy='auto', k_neighbors=5):
    """Apply SMOTE to oversample minority classes.
            - 'auto': Balance all classes to majority class count
            - dict: e.g., {0: 10000, 1: 5000, 2: 8000}
        k_neighbors: Number of neighbors for SMOTE
    """

    if not IMBLEARN_AVAILABLE:
        raise ImportError("imbalanced-learn not installed")
    
    logger.info("Applying SMOTE to training data...")
    logger.info(f"  Original distribution: {Counter(y_train)}")
    
    smote = SMOTE(
        sampling_strategy=strategy,
        k_neighbors=k_neighbors,
        random_state=42
    )
    
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    logger.info(f"  After SMOTE: {Counter(y_resampled)}")
    logger.info(f"  Generated {len(X_resampled) - len(X_train)} synthetic samples")
    
    return X_resampled, y_resampled


def apply_smote_tomek(X_train, y_train, strategy='auto'):
    """Apply SMOTE + Tomek Links (hybrid over/under-sampling).
    SMOTE creates synthetic minority samples.
    Tomek Links removes majority samples near decision boundary.
    """

    if not IMBLEARN_AVAILABLE:
        raise ImportError("imbalanced-learn not installed")
    
    logger.info("Applying SMOTE + Tomek Links...")
    logger.info(f"  Original distribution: {Counter(y_train)}")
    
    smt = SMOTETomek(
        sampling_strategy=strategy,
        random_state=42
    )
    
    X_resampled, y_resampled = smt.fit_resample(X_train, y_train)
    
    logger.info(f"  After SMOTE+Tomek: {Counter(y_resampled)}")
    logger.info(f"  Net change: {len(X_resampled) - len(X_train)} samples")
    
    return X_resampled, y_resampled


def apply_borderline_smote(X_train, y_train, strategy='auto'):
    """Apply Borderline-SMOTE (focus on hard examples).
    Only creates synthetic samples for minority instances
    near the decision boundary (harder to classify).
    """

    if not IMBLEARN_AVAILABLE:
        raise ImportError("imbalanced-learn not installed")
    
    logger.info("Applying Borderline-SMOTE...")
    logger.info(f"  Original distribution: {Counter(y_train)}")
    
    bsmote = BorderlineSMOTE(
        sampling_strategy=strategy,
        random_state=42
    )
    
    X_resampled, y_resampled = bsmote.fit_resample(X_train, y_train)
    
    logger.info(f"  After Borderline-SMOTE: {Counter(y_resampled)}")
    
    return X_resampled, y_resampled


def compute_sample_weights(y_train, strategy='balanced'):
    
    logger.info("Computing sample weights...")
    
    counts = Counter(y_train)
    total = len(y_train)
    
    if strategy == 'balanced':
        # Inverse frequency: weight = total / (n_classes * count)
        n_classes = len(counts)
        class_weights = {cls: total / (n_classes * count) 
                        for cls, count in counts.items()}
    
    elif strategy == 'sqrt':
        # Square root of inverse frequency (less aggressive)
        class_weights = {cls: np.sqrt(total / count) 
                        for cls, count in counts.items()}
    
    elif isinstance(strategy, dict):
        class_weights = strategy
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Map to sample weights
    sample_weights = np.array([class_weights[y] for y in y_train])
    
    logger.info(f"  Class weights: {class_weights}")
    
    return sample_weights


def get_balanced_strategy(y_train, target_ratio=0.5):

    counts = Counter(y_train)
    majority_count = max(counts.values())
    target_count = int(majority_count * target_ratio)
    
    # Only boost classes below target
    strategy = {cls: max(count, target_count) 
                for cls, count in counts.items()
                if count < target_count}
    
    logger.info(f"Balanced strategy (ratio={target_ratio}): {strategy}")
    
    return strategy


# Chosen approach
def recommended_balancing(X_train, y_train, method='smote_tomek'):
    
    if not IMBLEARN_AVAILABLE and method != 'none':
        logger.warning("imbalanced-learn not available, skipping resampling")
        return X_train, y_train
    
    if method == 'none':
        logger.info("No resampling - relying on class_weight='balanced'")
        return X_train, y_train
    
    elif method == 'smote_tomek':
        # Balance minority (Medium Rating) to 60% of majority
        strategy = get_balanced_strategy(y_train, target_ratio=0.6)
        return apply_smote_tomek(X_train, y_train, strategy=strategy)
    
    elif method == 'borderline':
        strategy = get_balanced_strategy(y_train, target_ratio=0.6)
        return apply_borderline_smote(X_train, y_train, strategy=strategy)
    
    elif method == 'mild_smote':
        strategy = get_balanced_strategy(y_train, target_ratio=0.5)
        return apply_smote(X_train, y_train, strategy=strategy)
    
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":

    print("Class Balancing Demo")
    print("=" * 60)

    np.random.seed(42)
    X = np.random.randn(1000, 5)
    y = np.array([0]*455 + [1]*208 + [2]*337)  # Matches your distribution
    
    print(f"\nOriginal distribution: {Counter(y)}")
    
    if IMBLEARN_AVAILABLE:
        # Test different methods
        X_res, y_res = recommended_balancing(X, y, method='smote_tomek')
        print(f"After SMOTE+Tomek: {Counter(y_res)}")
    else:
        print("\nInstall imbalanced-learn to test balancing methods:")
        print("  pip install imbalanced-learn")
