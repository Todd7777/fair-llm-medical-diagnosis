# using fine tuned model weights


import torch


model = models.efficientnet_b0(weights="DEFAULT")


# 2. Load the trained weights
model.load_state_dict(torch.load(f"cnn_weights/{name}_fine_tuned.pt"))

# 3. Set the model to evaluation mode
model.eval()

# 4. Move to CUDA if needed
# model.to('cuda')

# 5. Testing / inference
with torch.no_grad():
    for inputs, labels in test_dataloader:  # Replace with your DataLoader
        # inputs = inputs.to('cuda')  # if using GPU
        outputs = model(inputs)
        # evaluate outputs, e.g., get predictions
        # predicted = torch.argmax(outputs, dim=1)
        # compare predicted to labels, etc.
