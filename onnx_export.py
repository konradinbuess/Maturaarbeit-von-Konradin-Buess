#Der gesamte Code zur Modellerstellung wurde anhand einer Beschreibung des Modells durch Gemini erstellt.
import torch
import torch.nn as nn

class MatrixFusionModule(nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, chunk_matrix, chunk_boundary, chunk_counts):
        mask = torch.matmul(chunk_matrix, chunk_boundary) <= 0
        return torch.sum(torch.where(mask, chunk_counts, torch.tensor(0.0, dtype=torch.float32, device=chunk_matrix.device)), dim=1)

device = torch.device("cpu")

model = MatrixFusionModule().to(device)

dummy_matrix = torch.randn(500, 8, dtype=torch.float32, device=device)
dummy_boundary = torch.randn(8, 10000, dtype=torch.float32, device=device)
dummy_counts = torch.randn(10000, dtype=torch.float32, device=device)

# Modell exportieren
torch.onnx.export(
    model,
    (dummy_matrix, dummy_boundary, dummy_counts),
    "gpu_matrix_fusion.onnx",
    input_names=["chunk_matrix", "boundary_matrix", "vector_counts"],
    output_names=["partial_res"],
    dynamic_axes={
        "chunk_matrix": {0: "num_points"},
        "boundary_matrix": {1: "num_vectors"},
        "vector_counts": {0: "num_vectors"}
    },
    opset_version=14
)
print("ONNX-Modell erfolgreich exportiert: gpu_matrix_fusion_old.onnx")

