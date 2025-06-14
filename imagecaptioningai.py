import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import matplotlib.pyplot as plt

model_name = "nlpconnect/vit-gpt2-image-captioning"

print("Loading model...")
model = VisionEncoderDecoderModel.from_pretrained(model_name)
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_caption(image_path):
    i_image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=i_image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    output_ids = model.generate(
        pixel_values,
        attention_mask=attention_mask,
        max_length=16
    )
    preds = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return preds

if __name__ == "__main__":
    img_path = rC\User\Win11\Pictures\"Image.path"

    img = Image.open(img_path)
    plt.imshow(img)
    plt.axis("off")
    plt.show()

    caption = predict_caption(img_path)
    print(f"\nGenerated Caption: {caption}")