from transformers import ASTForAudioClassification

def create_model(num_labels):
    model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset", num_labels=num_labels)
    return model
