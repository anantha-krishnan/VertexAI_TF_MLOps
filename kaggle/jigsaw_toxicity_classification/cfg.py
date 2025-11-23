import sys
import os
from pathlib import Path
# 1. Force Legacy Mode
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# 2. Import tf_keras first
import tf_keras

# 3. THE TRICK: Tell Python that 'keras' is actually 'tf_keras'
# This forces keras_tuner to use tf_keras internally.
sys.modules["keras"] = tf_keras
import tf_keras as keras 
import tensorflow as tf

class CFG:
    current_dir = Path(__file__).resolve().parent
    BASE_PATH = current_dir / '../../data/jigsaw-toxic-comment-classification-challenge'  # Base path for data files
    seed = 42  # Random seed
    preset = "distilbert-base-uncased"#"roberta-base"# "deberta_v3_extra_small_en" # Name of pretrained models
    sequence_length = 256  # Input sequence length
    epochs = 10 # Training epochs
    batch_size = 2  # Batch size
    scheduler = 'cosine'  # Learning rate scheduler
    label_cols = [
        'toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']  # Target labels
    text_col = 'comment_text'  # Input text column
    metrics = [
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.AUC(name='auc', multi_label=True)
    ]  # Evaluation metrics
    shuffle = True  # Shuffle dataset
    alpha = 0.25  # Focal loss alpha parameter
    gamma = 2.0  # Focal loss gamma parameter
    n_splits = 5  # Number of folds for cross-validation
    learning_rate = 3e-5  # Learning rate
    weight_decay = 1e-6  # Weight decay
    warmup_ratio = 0.1  # Warmup ratio for learning rate scheduler
    max_grad_norm = 1.0  # Maximum gradient norm for clipping
    dropout_rate = 0.3  # Dropout rate for regularization
    hidden_size = 256  # Hidden layer size
    dense_size = 128  # Dense layer size
    tuner_epochs = 2  # Number of epochs for hyperparameter tuning
    tuner_batch_size = 8  # Batch size for hyperparameter tuning
    tuner_trials = 5  # Number of trials for hyperparameter tuning
    tuner_executions_per_trial = 1  # Executions per trial for hyperparameter tuning
    model_dir = './model_checkpoints'  # Directory to save model checkpoints
    submission_file = './submission.csv'  # Path to save submission file
    pretrained_dir = './pretrained_models'  # Directory to save pretrained models
    log_dir = './logs'  # Directory for TensorBoard logs
    use_amp = True  # Use Automatic Mixed Precision
    device = 'cuda' if tf.config.list_physical_devices('GPU') else 'cpu'  # Device configuration
    num_workers = 4  # Number of workers for data loading
    pin_memory = True  # Pin memory for data loading
    early_stopping_patience = 3  # Early stopping patience
    early_stopping_monitor = 'val_loss'  # Metric to monitor for early stopping
    early_stopping_mode = 'min'  # Mode for early stopping ('min' or 'max')
    early_stopping_restore_best_weights = True  # Restore best weights on early stopping
    random_state = 42  # Random state for reproducibility
    verbose = 1  # Verbosity level
    save_best_only = True  # Save only the best model
    save_weights_only = False  # Save the entire model, not just weights
    save_freq = 'epoch'  # Frequency to save the model
    monitor_metric = 'val_loss'  # Metric to monitor for saving the model
    mixed_precision = True 
    lora_r = 8                # Small rank is enough for classification
    lora_alpha = 2*lora_r           # Alpha = 2 * Rank
    lora_dropout = 0.1        # Slightly higher dropout prevents overfitting on toxic data
    target_modules = ["query", "key", "value", "dense"]
