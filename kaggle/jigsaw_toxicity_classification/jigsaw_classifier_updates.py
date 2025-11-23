import sys
import os

# 1. Force Legacy Mode
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# 2. Import tf_keras first
import tf_keras

# 3. THE TRICK: Tell Python that 'keras' is actually 'tf_keras'
# This forces keras_tuner to use tf_keras internally.
sys.modules["keras"] = tf_keras

# 4. NOW import the rest
import tensorflow as tf
import numpy as np
import keras_tuner as kt
from transformers import AutoTokenizer, TFAutoModel
# You can now refer to tf_keras simply as keras, or keep your alias
import tf_keras as keras 
from tf_keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns # Optional, just makes the colors prettier
from sklearn.model_selection import KFold
import tf_keras.backend as K # To clear memory
import gc # Garbage collector
# local imports
from cfg import CFG

class JigsawClassifier:
    def __init__(self, CFG=CFG):
        self.CFG = CFG
        self.train_df, self.test_combined_df, self.test_combined_cleaned_df = self.preprocess_input_csv()
        test_combined_cleaned_02 = self.test_combined_cleaned_df.sample(frac=0.2, random_state=42).reset_index(drop=True)
        if CFG.device == 'cpu':
            self.test_combined_cleaned_df = test_combined_cleaned_02
            self.steps_per_epoch = 100
            self.validation_steps = 20
        else:
            self.steps_per_epoch = len(self.train_df) // self.CFG.batch_size
            self.validation_steps = len(self.test_combined_cleaned_df) // self.CFG.batch_size
        self.model = self.build_model_tuner()
        self.compile_model()
        self.tokenizer = AutoTokenizer.from_pretrained(self.CFG.preset)
        self.train_ds, self.val_ds, self.test_ds = self.create_data_loader()

        # outputs
        self.y_pred = None


    def create_dataset(self, df, shuffle=None):
        if shuffle is None:
            shuffle = self.CFG.shuffle
        texts = df[self.CFG.text_col]
        labels = df[self.CFG.label_cols] if self.CFG.label_cols is not None else None
        encodings = self.tokenizer(
            texts.tolist(),
            truncation=True,
            padding='max_length',
            max_length=self.CFG.sequence_length,
            return_tensors='tf'
        )
        if labels is not None:
            dataset = tf.data.Dataset.from_tensor_slices((
                {
                    'input_ids': encodings['input_ids'],
                    'attention_mask': encodings['attention_mask']
                },
                labels.values
            ))
        else:
            dataset = tf.data.Dataset.from_tensor_slices({
                'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask']
            })
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(texts), seed=self.CFG.seed)
        dataset = dataset.batch(self.CFG.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    
    def preprocess_input_csv(self):
        if self.CFG.BASE_PATH is None:
            raise ValueError("BASE_PATH is not set in CFG.")
        train_df = pd.read_csv(f'{self.CFG.BASE_PATH}/train.csv')
        test_df = pd.read_csv(f'{self.CFG.BASE_PATH}/test.csv')
        test_labels_df = pd.read_csv(f'{self.CFG.BASE_PATH}/test_labels.csv')
        test_combined_df = pd.merge(test_df, test_labels_df, on='id')
        test_combined_cleaned_df = test_combined_df[~test_combined_df[self.CFG.label_cols].isin([-1]).any(axis=1)]
        return train_df, test_combined_df, test_combined_cleaned_df
    
    def build_model_tuner(self,hp=None):
        if hp is None:
            dropout_rate = self.CFG.dropout_rate
            learning_rate = self.CFG.learning_rate
        else:
            dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
            learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5])
        
        # Using 'keras' here (which is actually tf_keras now)
        input_ids = keras.layers.Input(shape=(self.CFG.sequence_length,), dtype=tf.int32, name='input_ids')
        attention_mask = keras.layers.Input(shape=(self.CFG.sequence_length,), dtype=tf.int32, name='attention_mask')
        
        base_model = TFAutoModel.from_pretrained(self.CFG.preset)
        output = base_model(input_ids, attention_mask=attention_mask)
        pooled_output = output.last_hidden_state[:, 0, :]
        
        dropout = keras.layers.Dropout(dropout_rate, name='dropout')(pooled_output)
        output = keras.layers.Dense(len(self.CFG.label_cols), activation='sigmoid', name='sigmoid_output')(dropout)
        
        model = keras.Model(inputs=[input_ids, attention_mask], outputs=output)
        return model
    def compile_model(self, hp=None):
        if hp is None:
            learning_rate = self.CFG.learning_rate
        else:
            learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5])
        
         # Using 'keras' here (which is actually tf_keras now)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            #loss='binary_crossentropy',
            loss=self.focull_loss,
            metrics=self.CFG.metrics
        )
        
    def early_stopping_callback(self):
        # Using 'keras' here (which is actually tf_keras now)
        return EarlyStopping(
            monitor=self.CFG.early_stopping_monitor,
            patience=self.CFG.early_stopping_patience,
            mode=self.CFG.early_stopping_mode,
            restore_best_weights=self.CFG.early_stopping_restore_best_weights
        )
    def get_tuner(self):
        tuner = kt.RandomSearch(
            hypermodel=self.build_model_tuner,
            objective=kt.Objective("val_loss", direction="min"),
            max_trials=2,
            executions_per_trial=1,
            overwrite=True,
            directory='kt_tuner_dir',
            project_name='jigsaw_classifier_tuning'
        )

        print("Search space summary:")
        tuner.search_space_summary()
        return tuner
    
    def create_data_loader(self):
        train_split_df, val_split_df = train_test_split(
            self.train_df,
            test_size=0.2,
            random_state=self.CFG.seed,
        )
        val_split_ds = self.create_dataset(
            val_split_df, shuffle=self.CFG.shuffle
        )
        train_split_ds = self.create_dataset(
            train_split_df, shuffle=self.CFG.shuffle)
        
        test_combined_cleaned_ds = self.create_dataset(
            self.test_combined_cleaned_df, shuffle=False    
        )
        return train_split_ds, val_split_ds, test_combined_cleaned_ds
    
    def train_model(self):
        early_stopping = self.early_stopping_callback()

        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.CFG.epochs,
            callbacks=[early_stopping],
            steps_per_epoch=self.steps_per_epoch,   
            validation_steps=self.validation_steps  
        )
        return history
    def predict(self):
        self.y_pred  = self.model.predict(self.test_ds)
        return self.y_pred
    def focull_loss(self, y_true, y_pred):
        """
        Compute the focal loss between `y_true` and `y_pred`.

        Args:
            y_true: Ground truth labels, shape of [batch_size, num_classes].
            y_pred: Predicted logits, shape of [batch_size, num_classes].
            alpha: Balancing factor.
            gamma: Focusing parameter.
        Returns:
            Focal loss value.
        Loss= #Part A (Positive Loss)+ Part B (Negative Loss)
                # Part A (Positive Loss)  -y_true ⋅log(y_pred) 
                # Part B (Negative Loss) (1-y_true)⋅log(1-y_pred)
        """
        alpha = self.CFG.alpha if hasattr(self.CFG, 'alpha') else 0.25
        gamma = self.CFG.gamma if hasattr(self.CFG, 'gamma') else 2.0
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
        p_t1 = tf.where(tf.equal(y_true, 1), y_pred, 1)
        p_t0 = tf.where(tf.equal(y_true, 0), y_pred, 0)
        # The Focal Loss Formula
        p = -tf.reduce_sum(alpha * tf.pow(1 - p_t1, gamma)* tf.math.log(p_t1))
        n = -tf.reduce_sum((1-alpha) * tf.pow(p_t0, gamma)* tf.math.log(1 - p_t0))
        return p + n

        alpha_factor = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        modulating_factor = tf.pow((1 - p_t), gamma)
        focal_loss = alpha_factor * modulating_factor * bce
        return tf.reduce_mean(focal_loss)
    
    def k_fold_model_training(self):
        # Implement k-fold cross-validation training here
        kf = KFold(n_splits=self.CFG.n_splits, shuffle=True, random_state=self.CFG.random_state)
        for fold, (train_index, val_index) in enumerate(kf.split(self.train_df)):
            K.clear_session()
            gc.collect()
            train_fold_df = self.train_df.iloc[train_index]
            val_fold_df = self.train_df.iloc[val_index]
            self.train_ds = self.create_dataset(train_fold_df, shuffle=self.CFG.shuffle)
            self.val_ds = self.create_dataset(val_fold_df, shuffle=self.CFG.shuffle)
            self.model = self.build_model_tuner()
            self.compile_model()
            early_stopping = self.early_stopping_callback()
            self.model.fit(
                self.train_ds,
                validation_data=self.val_ds,
                epochs=self.CFG.epochs,
                callbacks=[early_stopping],
                steps_per_epoch=self.steps_per_epoch,
                validation_steps=self.validation_steps
            )
            # After training on this fold, make predictions on the test set
            fold_y_pred = self.model.predict(self.test_ds)
            if self.y_pred is None:
                self.y_pred = fold_y_pred / self.CFG.n_splits
            else:
                self.y_pred += fold_y_pred / self.CFG.n_splits
            # --- Cleanup to save RAM ---
            del self.model, self.train_ds, self.val_ds
            
            
        
    def plot_metrics(self ):        
        # Set up the plot
        plt.figure(figsize=(10, 8))
        colors = sns.color_palette("bright", n_colors=len(CFG.label_cols))
        lw = 2 # Line width

        # Loop through each label (Toxic, Severe_Toxic, etc.)
        for i, label in enumerate(CFG.label_cols):
            # 1. Compute FPR and TPR for this specific label
            fpr, tpr, thresholds  = roc_curve(self.test_combined_cleaned_df[label].values, self.y_pred[:, i])
            J = tpr - fpr
            ix = np.argmax(J) # Index of the maximum J
            best_thresh = thresholds[ix]
            best_fpr = fpr[ix]
            best_tpr = tpr[ix]
            max_j = J[ix]
            print(f"{label:<15}: Best Thresh={best_thresh:.3f}, Max J={max_j:.3f}")
            # 2. Calculate the AUC score for this specific label
            roc_auc = auc(fpr, tpr)
            plt.scatter(best_fpr, best_tpr, color=colors[i], s=70, edgecolor='black', zorder=5)
            offset_y = -20 - (i * 12)
            # 3. Plot the curve
            plt.plot(fpr, tpr, color=colors[i], lw=lw,
                    label=f'{label} (area = {roc_auc:.2f})')
            plt.annotate(f'Th={best_thresh:.2f}', 
                        xy=(best_fpr, best_tpr), 
                        xytext=(20, offset_y), # Offset text to the right and down
                        textcoords='offset points',
                        fontsize=9, 
                        arrowprops=dict(arrowstyle="->", color='gray', alpha=0.5),
                        color=colors[i],
                        fontweight='bold')

        # Plot the "Random Guess" line (diagonal)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

        # Formatting
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('ROC Curves by Toxicity Type')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        plt.show()

