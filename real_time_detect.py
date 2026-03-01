import cv2
import time
import torch
from ultralytics import YOLO

# ---------------------------
# Load Model
# ---------------------------
model = YOLO("last.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

if device == "cuda":
    model.model.half()   # Faster GPU inference (FP16)

# # ---------------------------
# # Pest → Pesticide Dictionary
# # ---------------------------
pesticide_dict = {
        # RICE PESTS
    "rice leaf roller": "Chlorantraniliprole",
    "rice leaf caterpillar": "Spinosad",
    "paddy stem maggot": "Chlorpyrifos",
    "asiatic rice borer": "Cartap hydrochloride",
    "yellow rice borer": "Cartap hydrochloride",
    "rice gall midge": "Imidacloprid",
    "Rice Stemfly": "Chlorpyrifos",
    "brown plant hopper": "Buprofezin",
    "white backed plant hopper": "Thiamethoxam",
    "small brown plant hopper": "Imidacloprid",
    "rice water weevil": "Lambda-cyhalothrin",
    "rice leafhopper": "Imidacloprid",
    "grain spreader thrips": "Spinosad",
    "rice shell pest": "Malathion",
    # CUTWORMS / BORERS / LEPIDOPTERA
    "black cutworm": "Chlorpyrifos",
    "large cutworm": "Chlorpyrifos",
    "yellow cutworm": "Chlorantraniliprole",
    "army worm": "Emamectin benzoate",
    "corn borer": "Chlorantraniliprole",
# APHIDS
    "aphids": "Imidacloprid",
    "english grain aphid": "Thiamethoxam",
    "green bug": "Acetamiprid",
    "bird cherry-oataphid": "Imidacloprid",
    "Aphis citricola Vander Goot": "Imidacloprid",
    "Toxoptera citricidus": "Thiamethoxam",
    "Toxoptera aurantii": "Acetamiprid",
    # THRIPS
    "Thrips": "Spinosad",
    "wheat phloeothrips": "Spinosad",
    "Scirtothrips dorsalis Hood": "Spinosad",
    "odontothrips loti": "Spinosad",
    # MITES / ACARINES
    "red spider": "Abamectin",
    "longlegged spider mite": "Abamectin",
    "Panonchus citri McGregor": "Abamectin",
    "Brevipoalpus lewisi McGregor": "Abamectin",
    "Polyphagotars onemus latus": "Abamectin",
    # BEETLES / WEEVILS
    "flea beetle": "Carbaryl",
    "beet weevil": "Lambda-cyhalothrin",
    "alfalfa weevil": "Chlorpyrifos",
    "wireworm": "Fipronil",
    "white margined moth": "Chlorantraniliprole",
    # FRUIT / OTHER PESTS
    "Lycorma delicatula": "Imidacloprid",
    "Bactrocera tsuneonis": "Spinosad bait",
    "Dacus dorsalis(Hendel)": "Spinosad bait",
    "Phyllocnistis citrella Stainton": "Imidacloprid",
    "Papilio xuthus": "Chlorantraniliprole",
    # WHITEFLY / SAP SUCKERS
    "Trialeurodes vaporariorum": "Imidacloprid",
    "Aleurocanthus spiniferus": "Thiamethoxam",
    # SCALE / MEALYBUGS
    "Icerya purchasi Maskell": "Imidacloprid",
    "Ceroplastes rubens": "Imidacloprid",
    "Unaspis yanonensis": "Imidacloprid",
    # GENERAL LIST (remaining items)
    "Potosiabre vitarsis": "Lambda-cyhalothrin",
    "peach borer": "Chlorpyrifos",
    "green bug": "Acetamiprid",
    "english grain aphid": "Thiamethoxam",
    "sericaorient alismots chulsky": "Imidacloprid",
    "flax budworm": "Chlorantraniliprole",
    "alfalfa plant bug": "Thiamethoxam",
    "tarnished plant bug": "Imidacloprid",
    "Locustoidea": "Malathion",
    "lytta polita": "Carbaryl",
    "legume blister beetle": "Carbaryl",
    "blister beetle": "Carbaryl",
    "therioaphis maculata Buckton": "Imidacloprid",
    "alfalfa seed chalcid": "Chlorpyrifos",
    "Pieris canidia": "Chlorantraniliprole",
    "Apolygus lucorum": "Thiamethoxam",
    "Limacodidae": "Chlorantraniliprole",
    "Viteus vitifoliae": "Imidacloprid",
    "Colomerus vitis": "Abamectin",
    "oides decempunctata": "Lambda-cyhalothrin",
    "Polyphagotars onemus latus": "Lambda-cyhalothrin",
    "Pseudococcus comstocki Kuwana": "Imidacloprid",
    "parathrene regalis": "Fipronil",
    "Ampelophaga": "Fipronil",
    "Xylotrechus": "Fipronil",
    "Cicadella viridis": "Imidacloprid",
    "Miridae": "Thiamethoxam",
    "Erythroneura apicalis": "Imidacloprid",
    "Panonchus citri McGregor": "Oxamyl",
    "Phyllocoptes oleiverus ashmead": "Abamectin",
    "Icerya purchasi Maskell": "Imidacloprid",
    "Ceroplastes rubens": "Imidacloprid",
    "Chrysomphalus aonidum": "Imidacloprid",
    "Parlatoria zizyphus Lucus": "Imidacloprid",
    "Nipaecoccus vastalor": "Imidacloprid",
    "Tetradacus c Bactrocera minax": "Spinosad bait",
    "Dacus dorsalis(Hendel)": "Spinosad bait",
    "Prodenia litura": "Spinosad",
    "Adristyrannus": "Lambda-cyhalothrin",
    "Phyllocnistis citrella Stainton": "Imidacloprid",
    "Toxoptera citricidus": "Thiamethoxam",
    "Toxoptera aurantii": "Acetamiprid",
    "Aphis citricola Vander Goot": "Imidacloprid",
    "Scirtothrips dorsalis Hood": "Spinosad",
    "Dasineura sp": "Chlorpyrifos",
    "Lawana imitata Melichar": "Imidacloprid",
    "Salurnis marginella Guerr": "Lambda-cyhalothrin",
    "Deporaus marginatus Pascoe": "Lambda-cyhalothrin",
    "Chlumetia transversa": "Chlorantraniliprole",
    "Mango flat beak leafhopper": "Imidacloprid",
    "Rhytidodera bowrinii white": "Lambda-cyhalothrin",
    "Sternochetus frigidus": "Fipronil",
    "Cicadellidae": "Imidacloprid"
}
# # ---------------------------
# # Camera Setup
# # ---------------------------
# # Open camera
cap = cv2.VideoCapture(0)
# # Open camera stream (if using IP camera)

# # cap = cv2.VideoCapture('http://10.95.235.15:8080/video')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

prev_time = 0
frame_skip = 2
frame_count = 0

print("🚜 Smart High-Speed Pest Detection Started... Press Q to exit")

# # ---------------------------
# # Main Loop
# # ---------------------------
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    frame = cv2.resize(frame, (640, 480))

    # Inference (Fast Mode)
    with torch.no_grad():
        results = model(frame, conf=0.5, imgsz=640, verbose=False)

    annotated_frame = results[0].plot()

    pest_count = {}
    top_pesticide = None

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            pest_count[class_name] = pest_count.get(class_name, 0) + 1

            pesticide = pesticide_dict.get(class_name, "Consult Agri Officer")
            top_pesticide = pesticide

            print(f"⚠ Pest Detected: {class_name}")
            print(f"💊 Suggested: {pesticide}")
            print("------------------------")

#     # ---------------------------
#     # Show Pest Count
#     # ---------------------------
    y_offset = 30
    for pest, count in pest_count.items():
        cv2.putText(annotated_frame, f"{pest}: {count}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2)
        y_offset += 30

    # ---------------------------
    # Show Pesticide Suggestion
    # ---------------------------
    if top_pesticide:
        cv2.putText(annotated_frame,
                    f"Recommended: {top_pesticide}",
                    (10, annotated_frame.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2)

#     # ---------------------------
#     # FPS Counter
#     # ---------------------------
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    cv2.putText(annotated_frame,
                f"FPS: {int(fps)}",
                (10, annotated_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2)

    cv2.imshow("AgriGuard SmartDetect", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
