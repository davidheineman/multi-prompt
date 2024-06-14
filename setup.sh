# Create data folder
echo "Making data folders"
mkdir data && mkdir data/simplification && mkdir data/translation && mkdir data/code

# Download ALMA test data (repackaged WMT '21 and '22 data)
echo "Downloading ALMA test data..."
git clone https://github.com/fe1ixxu/ALMA.git
mv ALMA/human_written_data data/translation/alma
rm -rf ALMA

# Download NTREX additional reference data for 128 languages
echo "Downloading NTREX data..."
git clone https://github.com/MicrosoftTranslator/NTREX.git
mv NTREX/NTREX-128 data/translation/ntrex
rm -rf NTREX

# Download HumanEval
echo "Downloading HumanEval data..."
curl -O -L https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz
gunzip -c HumanEval.jsonl.gz > data/code/HumanEval.jsonl
rm HumanEval.jsonl.gz

# Download SimpEval
echo "Downloading SimpEval..."
git clone https://github.com/Yao-Dou/LENS.git
mv LENS/data data/simplification/lens-data
rm -rf LENS

# Install SLE
echo "Installing SLE..."
git clone https://github.com/liamcripwell/sle.git
mv sle src/metrics/sle
pip install -e sle
rm -rf sle

# Install MetricX
echo "Installing MetricX..."
git clone https://github.com/google-research/metricx.git
mv metricx/metricx23 src/metrics/metricx23
rm -rf metricx

# Install BART
echo "Installing BART..."
git clone https://github.com/neulab/BARTScore.git
mv BARTScore src/metrics/bart_score
gdown 1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m -O src/metrics/bart_score/bart-checkpoint.pth
rm -rf BARTScore
