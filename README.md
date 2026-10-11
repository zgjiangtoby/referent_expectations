# Referent expectations

The code for *Calibrated Referent Expectations and Subject Form in Three Languages*.

Build annotated subject opportunities, measure LLM referent expectations, and compare subject-form models with local-history and calibrated contextual features.

## Environment

Run the commands below in Bash. Use Python 3.11 and a CUDA GPU with enough memory for FP32 inference; Keep the batch settings below when comparing the reported results.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy==1.26.4 pandas==2.3.3 scipy==1.16.3 \
  transformers==4.57.2 tokenizers==0.22.1 accelerate==1.14.0 \
  safetensors==0.7.0 sentencepiece==0.2.1 huggingface-hub==0.36.0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p data models work
```

Use new output directories for a run. Inference writes score metadata beside the score file; analyses use those files and intermediate outputs from the earlier commands. Raw texts, labels, model weights and generated results are not included. Obtain the sources below under their own terms and retain their notices.

```bash
python -m pip install scikit-learn==1.7.2 udapi==0.5.2 termcolor==3.3.0 colorama==0.4.6
```

## CorefUD inputs

The study uses fixed native trees from UD Spanish-AnCora, Catalan-AnCora and Czech-PDTC. Preserve the native CoNLL-U text, document boundaries and `Entity` annotations. The trees below select the source bytes; a current default branch is not interchangeable.

```bash
git clone https://github.com/UniversalDependencies/UD_Spanish-AnCora.git data/es_repository
git clone https://github.com/UniversalDependencies/UD_Catalan-AnCora.git data/ca_repository
git clone https://github.com/UniversalDependencies/UD_Czech-PDTC.git data/cs_repository
mkdir -p data/es data/ca data/cs
git -C data/es_repository archive 20adddbfcdd773c6dc97ba48ea11ca9364e74185 | tar -x -C data/es
git -C data/ca_repository archive b27ff329b6b99ed40b99e3edb67f5de63b01248f | tar -x -C data/ca
git -C data/cs_repository archive 162cb9cb6e86a0d16c38a6e0fcbf8a3b4c241606 | tar -x -C data/cs
python prepare_corpus.py --ca-data data/ca --cs-data data/cs --es-data data/es \
  --output work/preparation
python prepare_entity_probe.py --measurement-root work/preparation/measurement \
  --analysis-root work/preparation/analysis --output work/probes
python diagnose_information_boundary.py --probe-items work/probes/items.jsonl \
  --source-rows-root work/preparation/analysis --output work/boundary
```

`prepare_corpus.py` lists the 15 required files and their byte sizes. Four Czech training shards have no coreference layer and contribute only to the inventory. UDAPI supplies mention boundaries and heads; no replacement parser or automatic relabeling is used. Probe preparation retains the original language order and document folds. It produces 154,698 prompts from 25,783 opportunities.

## Models and analyses

```bash
qwen_revision=b968826d9c46dd6066d109eabc6255188de91218
euro_revision=a25c7fa65fc2a644e6270b8940dbe295b51da681
hf download Qwen/Qwen3-8B --revision "$qwen_revision" --local-dir "models/$qwen_revision"
hf download utter-project/EuroLLM-1.7B-Instruct --revision "$euro_revision" --local-dir "models/$euro_revision"
```

Keep the revision hash as each model directory's basename: the comparisons check it against the score metadata. Inference uses only these local files.

```bash
for checkpoint in qwen euro; do
  if [ "$checkpoint" = qwen ]; then
    model="$qwen_revision"; model_name=Qwen3-8B; batch=8
  else
    model="$euro_revision"; model_name=EuroLLM-1.7B-Instruct; batch=32
  fi
  python score_choices.py --model "models/$model" --input work/probes/inputs.jsonl \
    --output "work/scores/$checkpoint.jsonl" --batch-size "$batch" --max-tokens 4096 \
    --dtype float32 --check-batch-equivalence --check-full-forward
  python analyze_probe.py --scores "work/scores/$checkpoint.jsonl" --refits 100 \
    --output "work/analysis_${checkpoint}_contexttokens"
  python analyze_local_history.py --analysis "work/analysis_${checkpoint}_contexttokens" \
    --boundary work/boundary --refits 100 --output "work/analysis_${checkpoint}_local_history"
  python analyze_calibrated_reference.py --scores "work/scores/$checkpoint.jsonl" \
    --reference-analysis "work/analysis_${checkpoint}_contexttokens" \
    --local-history "work/analysis_${checkpoint}_local_history" --boundary work/boundary \
    --model-name "$model_name" --output "work/analysis_${checkpoint}_calibrated_reference"
done
python compare_calibrated_reference.py --qwen work/analysis_qwen_calibrated_reference \
  --euro work/analysis_euro_calibrated_reference --output work/comparison_calibrated_reference
python analyze_mc_precision.py collect --work-dir work --output work/mc_draws --receipt work/mc_collection.json
python analyze_mc_precision.py report --input work/mc_draws --output work/mc_precision
```

The raw form, local-history and calibrated analyses keep their separate comparison families. Calibrated analyses use 2,000 conditional draws and 100 joint temperature/head refits. The final commands assess precision from these saved draws without fitting or resampling. Unbounded Monte Carlo sets for a percentile endpoint describe tail precision at 100 refits; they are not unbounded scientific-effect intervals. Outputs are CSV/JSON summaries plus the intermediate per-row and per-draw files needed by the next analysis.
