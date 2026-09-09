"""Render the repository-grounded methodology flowchart without ML execution."""
from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "research_methodology"
OUT.mkdir(parents=True, exist_ok=True)
W, H = 3300, 2630
im = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(im)
FONT = Path("C:/Windows/Fonts/arial.ttf")
BOLD = Path("C:/Windows/Fonts/arialbd.ttf")
ink, blue, gray, orange = "#172536", "#254A70", "#5C6873", "#946024"


def font(n, bold=False):
    return ImageFont.truetype(str(BOLD if bold else FONT), n)


def text(x, y, lines, size=33, bold=False, color=ink, leading=1.3):
    f = font(size, bold)
    for line in lines.split("\n"):
        d.text((x, y), line, font=f, fill=color, anchor="mt")
        y += int(size * leading)


def line(points, color=blue, dashed=False, width=5):
    if not dashed:
        d.line(points, fill=color, width=width, joint="curve")
        return
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        length = math.hypot(x2 - x1, y2 - y1)
        for t in range(0, int(length), 24):
            end = min(t + 13, length)
            d.line([(x1 + (x2-x1)*t/length, y1 + (y2-y1)*t/length),
                    (x1 + (x2-x1)*end/length, y1 + (y2-y1)*end/length)],
                   fill=color, width=width)


def arrow(points, dashed=False, color=blue):
    line(points, color, dashed)
    (x0, y0), (x, y) = points[-2:]
    a = math.atan2(y-y0, x-x0)
    d.polygon([(x, y), (x-20*math.cos(a-.48), y-20*math.sin(a-.48)),
               (x-20*math.cos(a+.48), y-20*math.sin(a+.48))], fill=color)


def box(cx, y, width, height, title, body="", pending=False):
    assert d.textlength(title, font=font(35, True)) < width - 24, title
    for row in body.split("\n"):
        assert d.textlength(row, font=font(28)) < width - 24, row
    assert 62 + max(0, len(body.split("\n"))-1)*32 + 28 <= height-8, title
    x = cx-width//2
    d.rectangle((x, y, x+width, y+height), fill="#FFF9F0" if pending else "#F3F6FA")
    line([(x,y),(x+width,y),(x+width,y+height),(x,y+height),(x,y)],
         orange if pending else blue, pending, 4)
    text(cx, y+20, title, 35, True, orange if pending else blue)
    if body:
        text(cx, y+62, body, 28, leading=1.15)


text(W//2, 48, "Three class acoustic fault classification", 56, True)
text(W//2, 123, "Base_de_Dados  |  Research methodology and evaluation boundaries", 32, color=gray)
box(1050, 215, 1100, 125, "Raw audio and integrity checks", "2,148 WAVs  |  12 folders  |  1 s mono at 44.1 kHz")
arrow([(1050,340),(1050,395)])
box(1050, 395, 1100, 135, "Assign three macro class labels", "Healthy: 179  |  Friction / wear: 1,074  |  Impulsive / structural: 895")
arrow([(1050,530),(1050,580)])
box(1050, 580, 1100, 130, "Frozen stratified recording level partition", "data/splits.csv  |  seed 42  |  not a verified session split")

for cx in (360, 1050, 1740):
    arrow([(1050,710),(1050,750),(cx,750),(cx,795)])
box(360,795,540,130,"Training recordings","n = 1,374  |  weight fitting")
box(1050,795,540,130,"Validation recordings","n = 344  |  model selection")
box(1740,795,540,130,"Test recordings","n = 430  |  assessment only")
for cx in (360,1050,1740):
    arrow([(cx,925),(cx,995)])
box(360,995,540,170,"Training representation","Noise / gain augmentation\nFixed log-Mel transform\nPer-recording normalization")
box(1050,995,540,170,"Validation representation","No augmentation\nSame fixed log-Mel transform\nPer-recording normalization")
box(1740,995,540,170,"Test representation","No augmentation\nSame fixed log-Mel transform\nPer-recording normalization")
arrow([(360,1165),(360,1230)])
box(360,1230,540,195,"TF-FaultNet model [A]","Architecture expanded at right\nThree logits; cross-entropy\nAdamW; 25 epochs\nTraining data only")
arrow([(1050,1165),(1050,1475)])
arrow([(630,1325),(710,1325),(710,1555),(780,1555)])
text(1220,1320,"Evaluate each epoch",28,color=gray)
box(1050,1475,540,160,"Validation evaluation","model.eval(); no gradients\nMonitor loss and macro F1")
arrow([(1050,1635),(1050,1710)])
box(1050,1710,540,160,"Lock selected checkpoint","Minimum validation loss\nHistorical selected epoch: 14")
arrow([(1050,1870),(1050,1950)])
arrow([(1740,1165),(1740,2030),(1450,2030)])
box(1050,1950,800,170,"Locked model test evaluation","Softmax probabilities and argmax\nConfusion matrix; macro F1; empirical ROC / PR")
arrow([(1050,2120),(1050,2190)])
box(1050,2190,800,130,"Historical report and error analysis","424 / 430 correct  |  accuracy 98.60%  |  macro F1 0.98589")

# Separate handcrafted branch: deliberately no connection into the CNN.
arrow([(90,860),(48,860),(48,1555),(90,1555)],True,orange)
box(360,1475,540,195,"Handcrafted EDA branch","Correct ZCR; training-only EDA\nThree-class descriptive plots\n25 EDA / 24 baseline features\nRegeneration still required",True)
arrow([(360,1670),(360,1735)],True,orange)
box(360,1735,540,195,"Baseline comparison needed","Majority; logistic regression\nRandom forest; XGBoost\nTrain-fitted preprocessing\nValidation-only selection",True)
arrow([(630,1810),(685,1810),(685,1650),(760,1650),(760,1595),(780,1595)],True,orange)
text(350,2000,"Handcrafted features are NOT\ninputs to TF-FaultNet",31,True,color=orange)

arrow([(1050,2320),(1050,2385)],True,orange)
box(1050,2385,1520,125,"Research completion and deployment validation still required",
    "Matched ablations  |  independent sessions / machines  |  robustness and monitoring",True)
text(1050,2545,"Solid: implemented / historical     Dashed: required research work     No test-to-training path",27,color=gray)

# The right panel expands the existing model, not a second learning branch.
line([(2130,215),(2130,2510)], "#D2D9E1", width=3)
text(2700,215,"[A] TF-FaultNet architecture",43,True,color=blue)
text(2700,274,"Same model used for training, validation and testing",28,color=gray)
model_stages = [
    (330,120,"Input waveform","B x 44,100 samples  |  mono  |  1 second"),
    (505,165,"Fixed log-Mel representation",
     "STFT: 1,024-point Hann window; hop 256; 64 Mel bands\nNatural log and per-recording normalization\nOutput: B x 1 x 64 x 173"),
    (715,165,"Convolutional stem",
     "Conv 5 x 5, stride 2, padding 2; 1 to 32 channels\nBatchNorm + ReLU; MaxPool 2 x 2, stride 2\nOutput: B x 32 x 16 x 43  |  864 parameters"),
    (925,165,"Residual block 1 with SE attention",
     "32 to 64 channels; 3 x 3 convolutions, strides 2 then 1\nSE channel gates; add projected shortcut; ReLU\nOutput: B x 64 x 8 x 22  |  58,752 parameters"),
    (1135,165,"Residual block 2 with SE attention",
     "64 to 128 channels; same downsampling block\nBatchNorm; SE; projected residual shortcut; ReLU\nOutput: B x 128 x 4 x 11  |  234,240 parameters"),
    (1345,165,"Residual block 3 with SE attention",
     "128 to 256 channels; same downsampling block\nBatchNorm; SE; projected residual shortcut; ReLU\nOutput: B x 256 x 2 x 6  |  935,424 parameters"),
    (1555,130,"Global pooling and concatenation",
     "Global average pool (256) + global maximum pool (256)\nConcatenate to B x 512  |  no trainable parameters"),
    (1740,165,"Dense classification head",
     "Dropout 0.35; Linear 512 to 256; ReLU\nDropout 0.20; Linear 256 to 3\nOutput: B x 3 logits  |  132,099 parameters"),
    (1950,130,"Training objective",
     "Raw logits feed cross-entropy with label smoothing 0.05\nNo manual softmax before CrossEntropyLoss"),
    (2135,120,"Prediction and reporting",
     "Logits -> softmax probabilities -> argmax class"),
    (2310,165,"Three mutually exclusive output classes",
     "0  Healthy baseline\n1  Continuous friction and wear\n2  Impulsive shocks and structural anomalies"),
]
for i, (y,h,title,body) in enumerate(model_stages):
    if i:
        prev_y,prev_h,*_ = model_stages[i-1]
        # Loss and prediction are separate uses of the head's logits.
        if i not in (8,9):
            arrow([(2700,prev_y+prev_h),(2700,y)])
    box(2700,y,1000,h,title,body)
# Objective is a side use of the logits; probabilities never come from loss.
arrow([(2700,1905),(2700,1950)])
arrow([(3200,1820),(3250,1820),(3250,2190),(3200,2190)])
text(2700,2500,"B = batch size  |  SE = squeeze-and-excitation channel attention",26,color=gray)
text(2700,2540,"1,361,379 trainable parameters  |  shapes and counts derived from source",25,color=gray)
text(2700,2580,"src/models.py and src/audio_features.py  |  no model weights changed",25,color=gray)
im.save(OUT / "three_class_research_flowchart.png", dpi=(240,240))
print(OUT / "three_class_research_flowchart.png")
