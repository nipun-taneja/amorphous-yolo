from ultralytics import YOLO
import wandb
from wandb.integration.ultralytics import add_wandb_callback

def main():
    wandb.init(project="amorphous-yolo-phase0", name="coco8_baseline")

    model = YOLO("yolov8n.pt") #tiny model
    add_wandb_callback(model)

    results = model.train(
        data="coco8.yaml",
        epochs = 3,
        imgsz = 640,
        project ="experiments",
        name="phase0_coco8_wandb",
        device="cpu"

    )

    wandb.finish()
    print("Training done:", results)


if __name__ == "__name__":
    main()