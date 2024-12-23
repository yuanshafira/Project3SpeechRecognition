# ASR Project Introduction and Objectives
## Introduction
        In the digital era, automatic speech recognition (ASR) has become a pivotal technology in voice-driven applications 
        like virtual assistants and customer support systems. As the demand for voice interfaces grows, achieving accurate 
        and efficient ASR is essential, especially for devices with limited resources. This project explores the integration 
        of pretrained ASR models, focusing on the Whisper-tiny model, to enhance speech recognition performance while 
        keeping resource requirements minimal.

        The Whisper-tiny model, developed by OpenAI, is known for its compact size and quick inference speed, making it 
        suitable for devices with restricted computational power. To test this model’s performance, we use the MINDS14 
        dataset by PolyAI, which includes diverse language and accent variations. This dataset offers a robust testing ground, 
        challenging the model to adapt to real-world speech nuances.

        Through this project, we aim to evaluate the effectiveness of the Whisper-tiny model in handling varied speech inputs 
        and to explore its potential for real-world applications in voice-driven systems.

## Objective

        The main objective of this research is to develop an ASR system tailored for the English language by using the 
        pretrained Whisper model. With the rich linguistic diversity of the MINDS14 dataset, we aim to:

        1. Improve speech recognition accuracy on devices with limited computational resources.
        2. Understand the strengths and limitations of pretrained ASR models in handling diverse language and accent variations.
        3. Evaluate the suitability of Whisper-tiny for deployment in resource-limited environments.

        This project seeks to advance ASR technology by showing the effectiveness of compact, efficient models in real-world 
        settings, making voice recognition solutions more accessible and adaptable.

## Loading and Merging the Dataset
The project uses the MINDS-14 dataset, specifically focusing on the US, Australian, and British English subsets, to assess the Whisper-tiny ASR model’s performance with varied English accents in the e-banking domain. The dataset is loaded through a structured MINDS14DataLoader class, which verifies the data’s integrity, checking for any missing or corrupted audio files and summarizing dataset statistics like sample count and intent distribution. The diversity of accents across these subsets helps train the model to better understand linguistic variations. After loading and verifying each subset, they are merged to increase the training data size, making the model more adaptable to different accents. This merging process includes checks for data loss, class balance, and consistency of features like transcriptions and audio properties. This approach aims to build a robust ASR model suited for real-world applications that encounter diverse English accents.

## Exploratory Data Analysis (EDA) Summary
1. Intent Class Distribution: A bar plot visualizes the frequency distribution of each intent class, providing insights into class balance. This ensures that the dataset covers a diverse range of intents, critical for effective model training.
2. Waveform and Spectrogram Analysis: For each intent, waveforms and spectrograms were generated to examine audio characteristics, such as amplitude changes and frequency components over time. This helps in understanding the variation in speech patterns across different intents.
3. Audio Feature Analysis with Chroma and FFT: Chroma (pitch class) and Fast Fourier Transform (FFT) analyses were conducted for selected intents. The chromagram reveals pitch variations, while FFT displays the magnitude spectrum, providing a detailed view of audio frequency components and aiding in feature extraction.
4. Null Value Check: The dataset was checked for null values in key features (e.g., audio, transcription, intent class) to ensure data completeness. No missing values were found, confirming the dataset's integrity.
5. Duplicate Analysis: Duplicate transcriptions were identified, helping to understand redundancy in data. While some duplication is acceptable, this analysis highlights repeated phrases that could potentially skew model training if overrepresented.

## Dataset Preprocessing
This preprocessing ensures that the data is clean, consistent, and suitable for training an ASR model, with a well-defined vocabulary and standardized audio format.

1. Dataset Splitting and Cleanup
- The dataset is divided into training and testing sets in an 80/20 ratio to ensure proper model evaluation.
- Unnecessary columns are removed to streamline the dataset for analysis.
- The structure of the dataset is verified to confirm that essential columns (e.g., transcription, intent class) are intact.
2. Transcription Verification
- Random samples of transcriptions are displayed to manually inspect the quality of text data.
- The audio and path columns are hidden to focus on transcription quality, helping identify any potential issues in text clarity or accuracy.
3. Text Cleaning
- Punctuation is removed, text is converted to lowercase, and extra spaces are standardized, ensuring uniformity across transcriptions.
- Samples of cleaned text are displayed to verify the effectiveness of the cleaning steps.

## Vocabulary Creation
# 6d. Create Vocabulary
```def extract_all_chars(batch):
    """
    Extract unique characters for vocabulary
    
    Args:
        batch: Dataset batch
    
    Returns:
        dict: Vocabulary and full text
    """
    all_text = " ".join(batch["transcription"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

# Extract vocabulary from both train and test sets
vocabs = merged_dataset.map(
    extract_all_chars,
    batched=True,
    batch_size=1,
    keep_in_memory=True,
    remove_columns=merged_dataset.column_names['train']
)

# Create vocabulary dictionary
vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))
vocab_dict = {v: k for k, v in enumerate(vocab_list)}
print(vocab_dict)

# Replace space with pipe symbol
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

# Add special tokens
vocab_dict["[UNK]"] = len(vocab_dict)  # Unknown token
vocab_dict["[PAD]"] = len(vocab_dict)  # Padding token
print(len(vocab_dict))
```

### Audio Standardization
- Since the MINDS-14 dataset uses an 8 kHz sampling rate, the audio files are converted to a 16 kHz sampling rate to match the input requirements of the Whisper model.
- The original sampling rate of each audio file is checked, then converted, and the result is verified to ensure consistency across the dataset.


### Dataset Extraction and Processor Testing
This extraction and processing pipeline ensures that the dataset is formatted and ready for ASR model training, with all audio and text data appropriately encoded for model input.

#### Processor Testing
1. Component Initialization: The smallest Whisper model (whisper-tiny) is used, with key components initialized:
- Tokenizer: Converts text into tokens.
- Feature Extractor: Processes audio into features suitable for the model.
- Processor: Combines both tokenizer and feature extractor to handle both text and audio inputs.
2. Sample Data Processing:
- A sample from the dataset is loaded, including audio data and its transcription.
- The audio is processed to generate input features (encoded_input), while the transcription is tokenized into input IDs (encoded_label).
- The raw audio data, original transcription, processed features, and encoded labels are printed for verification.
3. Model Prediction Testing:
- The model generates a prediction using the processed audio features.
- The prediction is decoded and compared to the true transcription.
- Result: The predicted transcription ("Freeze my card please.") closely matches the actual label ("freeze my card please"), confirming that the processor works correctly.

#### Feature Extraction for the Full Dataset
1. Batch Processing Function:
- The prepare_datasets function is created to process audio and transcription for each batch.
- It extracts audio features, tokenizes text, and adds the audio length in seconds.
2. Processing the Entire Dataset:
- The merged_dataset is processed in batches, using parallel processing (4 processes) to speed up the operation.
- Unnecessary columns are removed after processing to keep only relevant features (input_features, labels, input_length).
3. Output Verification:
- The processed dataset, now with input_features, labels, and input_length, is printed to confirm the structure.
- Optional checks are done to ensure consistency in the shapes of input features and labels.

### Whisper Model Building and Metrics
These steps finalize the Whisper model setup for ASR, using a robust set of metrics to evaluate model performance on both word and sentence levels, enhancing its adaptability and accuracy for real-world applications.
#### Custom Data Collator for Sequence-to-Sequence with Padding
1. Data Collator: A custom data collator, DataCollatorSpeechSeq2SeqWithPadding, is defined to handle batch processing for speech-to-text tasks.
2. Functionality:
- Pads audio features and text labels to ensure uniform sequence lengths within batches.
- Masks padding tokens with -100, which is ignored in the loss calculation, allowing the model to focus on actual transcriptions.
- Removes the beginning-of-sequence (BOS) token if present at the start of all sequences.

This collator helps manage variable-length audio and text sequences, making batch processing efficient and preparing data for model input.

#### Metrics
1. WER (Word Error Rate): A primary metric for ASR models, WER measures the percentage of words incorrectly predicted by comparing predictions with ground truth.
2. BLEU (Bilingual Evaluation Understudy Score): Calculates BLEU scores to evaluate the quality of generated text against references. Sentence-level BLEU scores are averaged across the dataset.
3. ROUGE (Recall-Oriented Understudy for Gisting Evaluation): Measures overlap of n-grams, specifically ROUGE-1 for unigram comparisons, assessing precision, recall, and F-score.
4. CER (Character Error Rate): Measures the proportion of character errors, useful for fine-grained error detection.
5. SER (Sentence Error Rate): Calculates the percentage of sentences that are entirely incorrect, indicating overall sentence-level accuracy.
6. F1 Score and R2 Score: The F1 score evaluates prediction accuracy for binary labels, while the R2 score measures correlation between predictions and true labels.
7. Accuracy: Computes the overall accuracy of predicted vs. reference transcriptions.
8. BLEU and ROUGE Scores: Additional metrics for evaluating sequence similarity, useful for fine-grained transcription evaluation.

The compute_metrics function combines these metrics, applying normalization and padding adjustments to ensure consistency and accuracy in evaluation.

#### Model Definition and Configuration
- Model Caching: The Whisper model configuration disables cache usage during training but re-enables it during generation to optimize memory and performance.
- Task-Specific Configuration: The model’s generate function is partially configured for English transcription tasks, with settings to improve inference efficiency and reduce computational load.

## Training Summary
### GPU Availability
The model is trained on an NVIDIA Tesla T4 GPU, which provides 14.75 GB of memory, enabling efficient training with the Whisper-tiny model.

### Training Configuration
        - Model: Whisper-tiny, optimized for ASR tasks.
        - Training Arguments:
                - Batch Size: 16 for training, with a gradient accumulation step of 1, allowing flexibility to adjust for memory constraints.
                - Learning Rate: Set to 3e-5 with a constant learning rate scheduler and warmup steps of 400.
                - Steps: Training runs for a maximum of 4000 steps, with evaluation and model saving every 400 steps.
                - Precision: Uses mixed-precision (fp16) for faster computations and memory efficiency.
                - Evaluation Strategy: Evaluates on the validation set every 400 steps, logging metrics for analysis.
                - Metrics: WER is used as the main metric to select the best model, and logging is enabled for TensorBoard visualization.
        - Trainer Setup: Configures the Seq2SeqTrainer with the specified model, dataset, custom data collator, and compute_metrics function to track performance.

```
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-tiny-minds14-english",
    # num_train_epochs=4,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=3e-5,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=400,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    fp16_full_eval=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=400,
    eval_steps=400,
    logging_steps=400,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
)

trainer = Seq2SeqTrainer(
    args= training_args,
    model=model,
    train_dataset=encoded_datasets['train'],
    eval_dataset=encoded_datasets['test'],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor,
    # optimizers=optimizer, # default optimizer is AdamW
)
```

### Training and Evaluation Results
The training logs display potential overfitting, possibly due to variations across the en-US, en-AU, and en-GB subsets. This variation might have impacted the model's ability to generalize across accents. Here’s a summary of performance at different stages:

<table>
  <thead>
    <tr>
      <th>Step</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Wer Ortho</th>
      <th>Wer</th>
      <th>Accuracy</th>
      <th>F1 Score</th>
      <th>R2 Score</th>
      <th>Bleu Score</th>
      <th>Rouge 1 r</th>
      <th>Rouge 1 p</th>
      <th>Rouge 1 f</th>
      <th>CER</th>
      <th>SER</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>400</td>
      <td>0.007000</td>
      <td>0.688022</td>
      <td>21.750256</td>
      <td>0.218353</td>
      <td>0.359116</td>
      <td>0.528455</td>
      <td>0.235858</td>
      <td>0.713570</td>
      <td>0.897348</td>
      <td>0.873094</td>
      <td>0.880645</td>
      <td>15.292417</td>
      <td>64.088398</td>
    </tr>
    <tr>
      <td>800</td>
      <td>0.017300</td>
      <td>0.723781</td>
      <td>21.989082</td>
      <td>0.221445</td>
      <td>0.348066</td>
      <td>0.516393</td>
      <td>0.489788</td>
      <td>0.711965</td>
      <td>0.893173</td>
      <td>0.870939</td>
      <td>0.877530</td>
      <td>15.500405</td>
      <td>65.193370</td>
    </tr>
    <tr>
      <td>1200</td>
      <td>0.009100</td>
      <td>0.758817</td>
      <td>21.238485</td>
      <td>0.213635</td>
      <td>0.348066</td>
      <td>0.516393</td>
      <td>0.244383</td>
      <td>0.713062</td>
      <td>0.898646</td>
      <td>0.874051</td>
      <td>0.882085</td>
      <td>14.809462</td>
      <td>65.193370</td>
    </tr>
    <tr>
      <td>1600</td>
      <td>0.004900</td>
      <td>0.765297</td>
      <td>23.404981</td>
      <td>0.236089</td>
      <td>0.359116</td>
      <td>0.528455</td>
      <td>0.456749</td>
      <td>0.717979</td>
      <td>0.888270</td>
      <td>0.878782</td>
      <td>0.878465</td>
      <td>17.439278</td>
      <td>64.088398</td>
    </tr>
    <tr>
      <td>2000</td>
      <td>0.004500</td>
      <td>0.813020</td>
      <td>22.773797</td>
      <td>0.229255</td>
      <td>0.334254</td>
      <td>0.501035</td>
      <td>0.550945</td>
      <td>0.699148</td>
      <td>0.882141</td>
      <td>0.870481</td>
      <td>0.871543</td>
      <td>15.937533</td>
      <td>66.574586</td>
    </tr>
    <tr>
      <td>2400</td>
      <td>0.002800</td>
      <td>0.829074</td>
      <td>21.750256</td>
      <td>0.218353</td>
      <td>0.328729</td>
      <td>0.494802</td>
      <td>0.273589</td>
      <td>0.710923</td>
      <td>0.881192</td>
      <td>0.870666</td>
      <td>0.871793</td>
      <td>15.655515</td>
      <td>67.127072</td>
    </tr>
    <tr>
      <td>2800</td>
      <td>0.005200</td>
      <td>0.855708</td>
      <td>23.251450</td>
      <td>0.234950</td>
      <td>0.348066</td>
      <td>0.516393</td>
      <td>0.577916</td>
      <td>0.715034</td>
      <td>0.871336</td>
      <td>0.873779</td>
      <td>0.865570</td>
      <td>17.118483</td>
      <td>65.193370</td>
    </tr>
    <tr>
      <td>3200</td>
      <td>0.004100</td>
      <td>0.845220</td>
      <td>26.953258</td>
      <td>0.268793</td>
      <td>0.353591</td>
      <td>0.522449</td>
      <td>-0.411504</td>
      <td>0.671624</td>
      <td>0.884932</td>
      <td>0.871449</td>
      <td>0.873303</td>
      <td>19.952762</td>
      <td>64.640884</td>
    </tr>
    <tr>
      <td>3600</td>
      <td>0.004300</td>
      <td>0.878660</td>
      <td>27.550324</td>
      <td>0.275952</td>
      <td>0.339779</td>
      <td>0.507216</td>
      <td>-0.055441</td>
      <td>0.681703</td>
      <td>0.862418</td>
      <td>0.873424</td>
      <td>0.860479</td>
      <td>20.188952</td>
      <td>66.022099</td>
    </tr>
    <tr>
      <td>4000</td>
      <td>0.003400</td>
      <td>0.866351</td>
      <td>21.443193</td>
      <td>0.215425</td>
      <td>0.375691</td>
      <td>0.546185</td>
      <td>0.562294</td>
      <td>0.719583</td>
      <td>0.886143</td>
      <td>0.879841</td>
      <td>0.878057</td>
      <td>15.436951</td>
      <td>62.430939</td>
    </tr>
  </tbody>
</table>

## Predict and Evaluate
### Plot Training Results
=== Best Metrics ===
Best WER: 21.24% (Step 1200)
Best CER: 14.81% (Step 1200)
Best BLEU: 0.718 (Step 1600)
Best Accuracy: 35.91% (Step 400)
Best F1 Score: 0.528 (Step 400)
Best ROUGE-F: 0.882 (Step 1200)
Best R2 Score: 0.578 (Step 2800)

=== Final Metrics (Step 3600) ===
Final WER: 27.55%
Final CER: 20.19%
Final SER: 66.02%
Final Accuracy: 33.98%
Final F1 Score: 0.507

### Predict and Evaluate
These metrics suggest that the Whisper-tiny model performs reasonably well on the test set, with an impressively low WER. The accuracy, F1, and BLEU scores indicate that while the model is effective, certain refinements—especially for sentence structure and context—could further enhance its reliability and precision.
=== Final Results ===
Average WER: 0.27%
Average Accuracy: 67.18%
Average F1 Score: 0.6778
Average BLEU Score: 0.5978

The results from this ASR (Automatic Speech Recognition) evaluation code show the system’s performance across multiple metrics on a sample of audio files.
=== Average Metrics ===
{'accuracy': 67.18208342636174,
 'bleu_score': 0.5977904802927451,
 'f1_score': 0.6777806557069361,
 'rouge_f1': 0.8089683728073176,
 'wer': 0.27155347148459097}

Insights:
- The relatively high ROUGE and BLEU scores indicate decent alignment between predicted and reference phrases.
- A WER of 27.16% suggests some room for improvement, especially in handling word-level accuracy.
- The F1 score, coupled with BLEU, implies that the model generally performs well but may miss some exact word matches or struggle with nuances in phrasing.


### Accuracy, WER (Word Error Rate), F1 score, and BLEU score from a list of prediction results
```

Processing 1 evaluation results...

=== Overall Metrics ===
Overall Accuracy: 67.18%
Overall WER: 0.27%
Overall F1: 0.68
Overall BLEU: 0.60

=== Additional Statistics ===
Accuracy Range: 67.18% - 67.18%
WER Range: 0.27% - 0.27%
F1 Range: 0.6778 - 0.6778
BLEU Range: 0.5978 - 0.5978

=== Sample Predictions ===

Prediction 1:
Target: sample target text
Predicted: sample predicted text
WER: 0.27%
Accuracy: 67.18%
F1 Score: 0.6778
```


## Recommendations
```

=== Model Performance Analysis ===

Current Metrics vs Targets:
ACCURACY: 67.18 ✗
WER: 0.27 ✗
F1: 0.68 ✗
BLEU: 0.60 ✗

=== Priority Actions ===
1. Improve accuracy by 12.82% through data augmentation and model scaling
2. Increase F1 score by 0.12 through dataset balancing and focal loss
3. Improve BLEU score by 0.10 through better language modeling
4. Reduce WER by 0.07% through enhanced preprocessing and custom loss functions

=== Detailed Recommendations ===

Data Augmentation:
- Add noise augmentation to training data
- Include more diverse accents and speaking styles
- Implement speed perturbation during training

Model Architecture:
- Consider increasing model size or using a larger pre-trained model
- Add custom task-specific layers for your domain

Training Strategy:
- Implement curriculum learning - start with cleaner, shorter utterances
- Use custom loss function weighted towards common error patterns
- Balance dataset across different utterance lengths
- Implement focal loss to handle class imbalance
- Add domain-specific pretraining step

Preprocessing:
- Improve audio preprocessing pipeline
- Implement better silence removal
- Add custom text normalization rules
```
