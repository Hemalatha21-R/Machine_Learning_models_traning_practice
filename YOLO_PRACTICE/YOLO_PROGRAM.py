from ultralytics import YOLO

# Step 1: Load YOLO model
model = YOLO("yolo26n.pt")

# Step 2: Give input image
results = model("yolo_image.jpg")

# Step 3: Display result
results[0].show()

# Step 4: Print detected objects
for result in results:

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        object_name = model.names[class_id]

        print(
            "Object:",
            object_name,
            "| Confidence:",
            round(confidence * 100, 2),
            "%"
        )

# Step 5: Save output
results[0].save(filename="output.jpg")

print("Detection completed successfully!")