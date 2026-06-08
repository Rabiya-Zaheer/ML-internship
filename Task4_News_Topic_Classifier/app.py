# ============================================================
#  News Topic Classifier — Gradio Deployment App
#  DevelopersHub Corp — AI/ML Internship Task 4
#
#  Run: python app.py
#  Then open the local URL (or share=True for public URL)
# ============================================================

import gradio as gr
from transformers import pipeline
import torch

# ── Load the fine-tuned model ────────────────────────────────────────────────
# Make sure you have fine-tuned the model first (run the notebook)
# The model folder is saved by: model.save_pretrained('./bert_news_model')

MODEL_PATH = './bert_news_model'   # change if saved elsewhere
LABELS     = ['World', 'Sports', 'Business', 'Sci/Tech']
LABEL_EMOJI = {'World': '🌍', 'Sports': '⚽', 'Business': '💼', 'Sci/Tech': '🔬'}

print(f"Loading model from {MODEL_PATH} ...")
classifier = pipeline(
    'text-classification',
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    return_all_scores=True,
    device=0 if torch.cuda.is_available() else -1  # use GPU if available
)
print("Model loaded ")


# ── Prediction function ──────────────────────────────────────────────────────
def classify_news(headline: str):
    """
    Takes a news headline string and returns a dict of label → confidence score.
    The Gradio Label component automatically shows a bar chart.
    """
    if not headline or not headline.strip():
        return {f"{LABEL_EMOJI[l]} {l}": 0.0 for l in LABELS}

    # Truncate to BERT's max length
    results = classifier(headline[:512])[0]

    # Format output with emoji labels
    scores = {
        f"{LABEL_EMOJI[r['label']]} {r['label']}": round(r['score'], 4)
        for r in results
    }
    return scores


# ── Example headlines ────────────────────────────────────────────────────────
examples = [
    ["Apple unveils M4-powered MacBook with on-device AI at WWDC 2025"],
    ["Manchester City defeats Real Madrid 3-1 in Champions League Final"],
    ["Federal Reserve raises interest rates for third consecutive time"],
    ["UN Security Council holds emergency meeting over Middle East tensions"],
    ["NASA's Artemis III crew successfully lands near lunar south pole"],
    ["Tesla reports record quarterly earnings, shares surge 15%"],
    ["Scientists discover new exoplanet that could support liquid water"],
    ["World Cup 2026: Argentina vs France preview and predictions"],
]


# ── Gradio Interface ─────────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft(), title="News Topic Classifier") as demo:

    gr.Markdown("""
    #  News Topic Classifier
    **Fine-tuned BERT model** for classifying news headlines into 4 categories.
    
    Trained on the [AG News dataset](https://huggingface.co/datasets/ag_news) 
    using [Hugging Face Transformers](https://huggingface.co/docs/transformers).
    """)

    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                label="News Headline",
                placeholder="Paste or type a news headline here...",
                lines=3
            )
            with gr.Row():
                btn_clear  = gr.Button("Clear")
                btn_submit = gr.Button("Classify →", variant="primary")

        with gr.Column(scale=2):
            output_label = gr.Label(
                label="Topic Probabilities",
                num_top_classes=4
            )

    gr.Examples(
        examples=examples,
        inputs=input_text,
        label="Try these examples"
    )

    gr.Markdown("""
    ---
    **Model**: `bert-base-uncased` fine-tuned on AG News  
    **Accuracy**: ~94% | **F1 Macro**: ~0.94  
    **Classes**: 🌍 World · ⚽ Sports · 💼 Business · 🔬 Sci/Tech
    """)

    btn_submit.click(fn=classify_news, inputs=input_text, outputs=output_label)
    btn_clear.click(fn=lambda: ("", None), inputs=[], outputs=[input_text, output_label])
    input_text.submit(fn=classify_news, inputs=input_text, outputs=output_label)


# ── Launch ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        share=True,          # set False for local-only
        server_name="0.0.0.0",
        server_port=7860,
    )
