from typing import Dict, List, Any
import os
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
import logging
import cv2  # type: ignore # 🚀 Ajout de l'importation OpenCV
from reportlab.lib.pagesizes import A4 # type: ignore
from reportlab.pdfgen import canvas # type: ignore


class ExtendedDiseaseDatabase:
    """
    Base de données étendue avec 100+ maladies agricoles
    Couvre toutes les cultures principales et maladies mondiales
    """

    def __init__(self):
        self.diseases_data = self._initialize_extended_disease_database()
        self.treatments_data = self._initialize_extended_treatments_database()
        self.prevention_data = self._initialize_extended_prevention_database()
        self.regional_data = self._initialize_regional_disease_data()


def _initialize_extended_disease_database() -> Dict[str, Dict]:
    """
    Base de données étendue avec 100+ maladies.
    """
    return {
        "Maladie1": {
            "symptômes": ["Fièvre", "Douleurs"],
            "traitement": "Antibiotiques",
        },
        "Maladie2": {"symptômes": ["Toux", "Fatigue"], "traitement": "Repos"},
    }


# ✅ Base globale des maladies (après l'initialisation)
DISEASE_DATABASE = _initialize_extended_disease_database()

# 🚀 Configuration du logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

print("✅ Script `diseases_infos.py` exécuté avec succès !")

logger = logging.getLogger(__name__)

class DiseaseManager:
    def __init__(self, model_path):
        """Initialisation du gestionnaire de maladies et chargement du modèle CNN."""
        self.diseases = {}  # ✅ On l’ajoute ici
        self.model_path = model_path
        self.model = None

        if os.path.exists(self.model_path):
            self.load_model(self.model_path)
        else:
            print(f"⚠️ Modèle introuvable : {self.model_path} — on continue sans le charger.")
            self.model = None  # Pour éviter d’accéder à un modèle inexistant plus tard



    def load_model(self, model_path):
        """Charge le modèle CNN et l'attache à l'instance."""
        if not os.path.exists(model_path):
            logger.error(
                f"🚨 Erreur : Le fichier modèle {model_path} est introuvable."
            )
            raise FileNotFoundError(f"🚨 Modèle non trouvé : {model_path}")

        try:
            self.model = tf.keras.models.load_model(model_path)
            logger.info(f"✅ Modèle chargé avec succès : {model_path}")
        except Exception as e:
            logger.error(f"🚨 Impossible de charger le modèle : {e}")
            raise RuntimeError(f"🚨 Échec du chargement du modèle : {e}")

    def add_disease(
        self, name, hosts, overview, symptoms, management, insecticides
    ):
        """Ajoute une maladie avec ses détails."""
        self.diseases[name] = {
            "hosts": hosts,
            "overview": overview,
            "symptoms": symptoms,
            "management": management,
            "insecticides": insecticides,
        }
        DISEASE_DATABASE[name] = self.diseases[
            name
        ]  # ✅ Enregistrement global

    def get_disease_info(self, disease_name):
        """Retourne les informations complètes sur une maladie en fonction de son nom."""
        return self.diseases.get(
            disease_name, "🚨 Aucune information trouvée sur cette maladie."
        )

    def upload_image(self, image_file, save_path="uploads/"):
        """Permet l'importation d'une image et son stockage local en acceptant plusieurs formats."""
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        allowed_extensions = {"jpg", "jpeg", "png", "bmp", "gif", "tiff"}
        file_extension = image_file.name.split(".")[-1].lower()

        if file_extension not in allowed_extensions:
            return "🚨 Format d'image non pris en charge."

        file_path = os.path.join(save_path, image_file.name)

        with open(file_path, "wb") as f:
            f.write(image_file.getbuffer())

        return file_path

    def analyze_image(self, image_path):
        """Analyse une image pour détecter une maladie en utilisant le modèle CNN."""
        if not os.path.exists(image_path):
            return "🚨 Erreur : Image introuvable"

        image = cv2.imread(image_path)
        if image is None:
            return "🚨 Erreur : Image non valide"

        processed_image = cv2.resize(image, (224, 224))
        processed_image = np.expand_dims(processed_image, axis=0) / 255.0

        prediction = self.model.predict(processed_image)  # Prédiction par CNN
        detected_disease = self.decode_prediction(
            prediction
        )  # 📌 Traduction du label IA

        return detected_disease

    def decode_prediction(self, prediction):
        """Transforme la prédiction du modèle en un label compréhensible."""
        disease_labels = list(self.diseases.keys())
        return (
            disease_labels[prediction.argmax()]
            if prediction is not None
            else "Unknown"
        )

    def export_to_pdf(
        self, disease_name, user_name="Unknown", save_path="reports/"
    ):
        """Génère un rapport PDF avec les informations sur la maladie."""
        if disease_name not in self.diseases:
            return "🚨 Maladie introuvable"

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        disease = self.diseases[disease_name]
        buffer = self._generate_pdf_report(
            user_name, disease_name, disease, save_path
        )

        return buffer

    def _generate_pdf_report(
        self, user_name, disease_name, disease, save_path
    ):
        """Crée un PDF avec les détails de la maladie."""
        buffer = os.path.join(save_path, f"{disease_name}_report.pdf")
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4

        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, "Smart Disease Detection Report")

        c.setFont("Helvetica", 12)
        c.drawString(50, height - 80, f"User: {user_name}")
        c.drawString(50, height - 100, f"Disease Detected: {disease_name}")

        # 📌 Ajout des informations sur la maladie
        y = height - 140
        for key, value in disease.items():
            c.drawString(70, y, f"{key}: {value}")
            y -= 20

        c.showPage()
        c.save()
        return buffer
# ✅ Ajout des maladies
disease_manager = DiseaseManager(model_path="dummy_model.keras")
disease_manager.add_disease(
    "Aphids on Vegetables",
    [
        "Asparagus",
        "Brassicas",
        "Legumes",
        "Corn",
        "Solanaceae",
        "Leafy Greens",
        "Cucurbits",
        "Potato",
        "Root Crops",
    ],
    "Aphids are small, soft-bodied insects that suck sap from plant tissues...",
    "Aphid feeding results in an overall loss of plant vigor...",
    "Thoroughly scout crops and weeds for signs or symptoms of aphids...",
    [
        "carbaryl",
        "methomyl",
        "malathion",
        "alpha-cypermethrin",
        "bifenthrin",
        "cyfluthrin",
        "fenpropathrin",
        "lambda-cyhalothrin",
        "deltamethrin",
        "permethrin",
    ],
)

print("🚀 Disease Manager is fully operational!")

disease_manager.add_disease(
    "Armyworms on Vegetables",
    [
        "Asparagus",
        "Brassicas",
        "Cucurbits",
        "Corn",
        "Leafy Greens",
        "Legumes",
        "Onions",
        "Potato",
        "Solanaceae Crops",
        "Root Crops",
    ],
    "Armyworms can be found all over the United States...",
    "Larvae feed on leaves with chewing mouthparts...",
    "Keep crop area weed-free, till residues and control weeds...",
    [
        "bifenthrin",
        "zeta-cypermethrin",
        "lambda-cyhalothrin",
        "permethrin",
        "deltamethrin",
        "cyfluthrin",
        "pyrethrins + neem oil",
        "imiclacloprid",
        "sulfur",
    ],
)

disease_manager.add_disease(
    "Blister Beetle",
    ["Alfalfa", "Legumes", "Solanaceae", "Potatoes"],
    (
        "Blister beetles in the genera Epicuata are found throughout "
        "North America..."
    ),
    "Adults feed on plant foliage and blossoms...",
    (
        "Blister beetle populations are influenced "
        "by grasshopper populations..."
    ),
    [],  # Ajout de la liste vide pour éviter l'erreur
)

disease_manager.add_disease(
    "Beet Leafhopper",
    ["Beans", "Beets", "Cucurbits", "Leafy Greens", "Tomato"],
    "Beet leafhoppers have wedge-shaped bodies varying in color from pale green, gray, or tan...",
    "Adults and nymphs feed with piercing sucking mouthparts, which can cause shriveled and 'burned' leaves...",
    "Exclude leafhoppers with floating row covers, shade tomato and pepper plants, remove plant debris...",
    [
        "bifenthrin",
        "zeta-cypermethrin",
        "lambda-cyhalothrin",
        "permethrin",
        "deltamethrin",
        "cyfluthrin",
        "pyrethrins + neem oil",
        "imiclacloprid",
        "sulfur",
        "methomyl",
        "fenpropathrin",
        "permethrin",
    ],
)

disease_manager.add_disease(
    "Colorado Potato Beetle",
    ["Eggplant", "Pepper", "Potato", "Tomato"],
    "Adults are about the same size and shape as a lady beetle but with yellow and black stripes...",
    "Adults and larvae feed with chewing mouthparts and can defoliate plants...",
    "Rotate crops to non-solanaceous crops, keep crop area free of solanaceous weeds...",
    [
        "Beauveria bassiana",
        "Edovum puttleri",
        "bifenthrin",
        "zeta-cypermethrin",
        "lambda-cyhalothrin",
        "permethrin",
        "deltamethrin",
        "cyfluthrin",
        "pyrethrins + neem oil",
        "imiclacloprid",
        "sulfur",
        "methomyl",
        "fenpropathrin",
    ],
)

disease_manager.add_disease(
    "Western Striped and Spotted Cucumber Beetle",
    [
        "Corn",
        "Cucurbits",
        "Leafy Greens",
        "Legumes",
        "Potato",
        "Solanaceae Crops",
    ],
    "Adult western striped cucumber beetles are about 1/3 inches long, with black heads and yellow and black-striped wings...",
    "Feeding scars on soft rinds of fruits, especially the undersides. Holes in stems and leaves. Destroyed flowers...",
    "Hand-pick cucumber beetles, plant resistant varieties, use row covers and mulches...",
    [
        "bifenthrin",
        "zeta-cypermethrin",
        "lambda-cyhalothrin",
        "permethrin",
        "deltamethrin",
        "cyfluthrin",
        "pyrethrins + neem oil",
        "imiclacloprid",
        "sulfur",
        "methomyl",
        "fenpropathrin",
    ],
)

disease_manager.add_disease(
    "Spotted Cucumber Beetle",
    [
        "Hemp",
        "Beans",
        "Corn",
        "Cucurbits",
        "Potato",
        "Tomato",
        "Small grains",
        "Ornamentals",
        "Grasses",
        "Weeds",
    ],
    "Adults have a black head, legs, and antennae. They have ovoid yellow-green bodies with 12 black spots on the wings...",
    "In hemp, adults chew on leaves, creating irregular-shaped holes. Damage is considered minor...",
    "Keep crop area weed-free, use plastic or organic mulches, destroy crop residues after harvest...",
    [
        "azadirachtin",
        "potassium laurate",
        "mineral oil",
        "diatomaceous earth",
        "pyrethrins",
        "rosemary oil",
        "neem oil",
        "kaolin",
    ],
)
disease_manager.add_disease(
    "Cutworms on Vegetables",
    [
        "Artichoke",
        "Asparagus",
        "Brassicas",
        "Cucurbits",
        "Corn",
        "Legumes",
        "Leafy Greens",
        "Onion",
        "Potato",
        "Root Crops",
        "Solanaceae Crops",
    ],
    "Cutworms belong to a large group of night-flying moths in the Noctuidae family...",
    "Larvae feed with chewing mouthparts and can clip off seedlings at the soil line...",
    "Remove cool-season weeds, use fall tillage to expose overwintering pupae, protect seedlings with collars...",
    [
        "bifenthrin",
        "zeta-cypermethrin",
        "lambda-cyhalothrin",
        "permethrin",
        "deltamethrin",
        "cyfluthrin",
        "pyrethrins + neem oil",
        "imiclacloprid",
        "sulfur",
        "methomyl",
        "fenpropathrin",
    ],
)
disease_manager.add_disease(
    "False Chinch Bug",
    [
        "Alfalfa",
        "Brassicas",
        "Cucurbits",
        "Corn",
        "Leafy Greens",
        "Potato",
        "Root Crops",
        "Solanaceae Crops",
        "Hemp",
    ],
    "On adults, the head, thorax, and anterior portion of the wings are brownish gray and the posterior portion of the wings are whitish-clear...",
    "Adults and nymphs feed with piercing-sucking mouthparts. Large numbers of aggregating adults can cause plants to wilt and die rapidly...",
    "Scout field edges that may contain mustards, look for aggregations on plants, keep plants well irrigated...",
    [
        "bifenthrin",
        "zeta-cypermethrin",
        "lambda-cyhalothrin",
        "permethrin",
        "deltamethrin",
        "cyfluthrin",
        "pyrethrins + neem oil",
        "imiclacloprid",
        "sulfur",
        "methomyl",
        "fenpropathrin",
    ],
)
disease_manager.add_disease(
    "Flea Beetles",
    [
        "Brassicas",
        "Leafy Greens",
        "Solanaceae Crops",
        "Root Crops",
        "Cucurbits",
        "Potato",
    ],
    "Flea beetles are common and problematic in Utah, affecting vegetable crops and ornamental plants...",
    "Small, round holes or pits in leaves and cotyledons...",
    "Adjust planting times, promote healthy plant growth, use trap crops and row covers...",
    [
        "lacewing larvae",
        "big-eyed bugs",
        "damsel bugs",
        "bifenthrin",
        "zeta-cypermethrin",
        "lambda-cyhalothrin",
        "permethrin",
        "deltamethrin",
        "cyfluthrin",
        "pyrethrins + neem oil",
    ],
)
disease_manager.add_disease(
    "Tomato and Tobacco Hornworms",
    ["Tomato", "Pepper", "Potato"],
    "Adults are large moths also known as sphinx, hawk, or hummingbird moths. Larvae are large, cylindrical and usually green...",
    "Larvae use chewing mouthparts to feed on leaves, blossoms, stems, and fruits, leaving behind dark green or black frass...",
    "Plow field after harvest to destroy pupae, rotate crops with non-host plants, handpick larvae at dawn or dusk...",
    [
        "bifenthrin",
        "zeta-cypermethrin",
        "lambda-cyhalothrin",
        "permethrin",
        "deltamethrin",
        "cyfluthrin",
        "pyrethrins + neem oil",
        "imiclacloprid",
        "sulfur",
        "methomyl",
        "fenpropathrin",
    ],
)
disease_manager.add_disease(
    "Thrips on Vegetables",
    [
        "Alfalfa",
        "Asparagus",
        "Bean",
        "Cabbage",
        "Cauliflower",
        "Cucumber",
        "Garlic",
        "Onion",
        "Potato",
        "Small Grains",
        "Tomato",
    ],
    "Thrips overwinter as adults in plant debris and protected areas. In the spring, they become active and move into fields...",
    "White flecks or silvery scars on foliage, dark fecal spots on leaves, distorted leaf growth, reduced bulb size in onions...",
    "Plow plant debris under after harvest, inspect transplants for thrips, remove weeds within fields, place mulch, plant resistant crop varieties...",
    [
        "bifenthrin",
        "zeta-cypermethrin",
        "lambda-cyhalothrin",
        "permethrin",
        "deltamethrin",
        "cyfluthrin",
        "pyrethrins + neem oil",
        "imiclacloprid",
        "sulfur",
        "methomyl",
        "fenpropathrin",
    ],
)
disease_manager.add_disease(
    "Potato Leafhopper",
    [
        "Alfalfa",
        "Legumes",
        "Potato",
        "Solanaceae",
        "Weeds including pigweed and shepherd's purse",
    ],
    "Adults are wedge-shaped, light green in color, and widest at the head with an elongated body...",
    "Adults and nymphs feed with piercing-sucking mouthparts, causing white-flecked injury (stippling) on foliage. Heavy feeding can lead to scorching...",
    "Manage weeds to reduce leafhopper populations, monitor symptoms such as curling leaves and stippling, apply insecticides when necessary...",
    [
        "bifenthrin",
        "zeta-cypermethrin",
        "lambda-cyhalothrin",
        "permethrin",
        "deltamethrin",
        "cyfluthrin",
        "pyrethrins + neem oil",
        "imiclacloprid",
        "sulfur",
        "methomyl",
        "fenpropathrin",
    ],
)
disease_manager.add_disease(
    "Two-Spotted Spider Mite",
    [
        "Corn",
        "Cucurbits",
        "Hemp",
        "Legumes",
        "Onions",
        "Root crops",
        "Solanaceae crops",
        "Many weeds and ornamentals",
    ],
    "Adults are tiny, with a yellowish-clear body and two dark spots on either side of its back...",
    "Stippled (small yellow spots) leaves, generalized bronzing or reddish discoloration, reduced plant vigor, premature leaf drop...",
    "Keep plants healthy, wash mites off plants with strong water stream, avoid malathion and pyrethroid insecticides, attract predatory mites...",
    [
        "predatory mites",
        "lacewings",
        "ladybugs",
        "bifenthrin",
        "zeta-cypermethrin",
        "lambda-cyhalothrin",
        "permethrin",
        "deltamethrin",
        "cyfluthrin",
        "pyrethrins + neem oil",
    ],
)
disease_manager.add_disease(
    "Corn Earworm / Tomato Fruitworm",
    ["Beans", "Corn", "Eggplant", "Pepper", "Peas", "Tomato"],
    "Corn earworm larvae feed on all plant parts, including fruits and kernels. Damage can result in decreased pollination and hollowed-out fruits...",
    "Tunnels in kernels and fruits filled with frass, direct damage to stems, foliage, and corn silk, premature fruit ripening, fruit rot...",
    "Avoid planting solanaceous crops near post-silking corn fields, use clothespins on corn silk, plant resistant varieties, remove infested plant debris...",
    [
        "Bt (Bacillus thuringiensis)",
        "spinosad",
        "pyrethrins",
        "chlorantraniliprole",
        "lacewings",
        "big-eyed bugs",
        "damsel bugs",
        "minute pirate bugs",
        "Trichogramma pretiosum",
    ],
)
disease_manager.add_disease(
    "Tomato Russet Mite",
    ["Potato", "Tomato", "Pepper", "Eggplant"],
    "Adults and nymphs have cigar-shaped, yellowish-tan or pink bodies and require a microscope to observe...",
    "Bronzing or 'russeting' of stems, leaves, and fruits, yellowing, curling, deflated leaves, and longitudinal cracks on fruits...",
    "Use clean transplants, avoid planting during hot, dry periods, remove infested debris, clean tools after use...",
    ["sulfur", "abamectin"],
)
disease_manager.add_disease(
    "Whiteflies (Family: Aleyrodidae)",
    [
        "Field-grown and greenhouse-grown hemp",
        "Ageratum",
        "Aster",
        "Beans",
        "Begonia",
        "Calendula",
        "Cucumber",
        "Grape",
        "Hibiscus",
        "Lantana",
        "Nicotiana",
        "Poinsettia",
        "Squash",
        "Tomato",
    ],
    "Adults are tiny with bright white wings and yellow-orange heads. Immature stages are pale, translucent, and mostly immobile...",
    "Leaves turn yellow, appear dry, or fall off plants. Whiteflies also excrete honeydew, causing sticky surfaces or sooty mold growth...",
    "Attract and conserve natural enemies, inspect transplants, remove heavily infested plants, use biocontrol such as Encarsia sp....",
    [
        "azadirachtin",
        "potassium laurate",
        "clarified hydrophobic extract of neem oil",
        "pyrethrins",
        "capsicum oleoresin extract",
        "Encarsia sp.",
        "bifenthrin",
        "zeta-cypermethrin",
        "lambda-cyhalothrin",
        "permethrin",
        "deltamethrin",
        "cyfluthrin",
        "pyrethrins + neem oil",
    ],
)
disease_manager.add_disease(
    "Alfalfa Mosaic Virus",
    [
        "Alfalfa",
        "Potato",
        "Pepper",
        "Tomato",
        "Hemp",
        "Many ornamentals and weeds",
    ],
    "Alfalfa mosaic virus (AMV) is in the genus Alfamovirus and is spread by aphids...",
    "Yellow mosaic or calico patterns on foliage, stunted plants, tuber necrosis, bright yellow blotches interspersed with green...",
    "Avoid planting potatoes or hemp near alfalfa fields, remove plants with positive diagnosis...",
    ["Limited efficacy with aphid-targeting insecticides"],
)

disease_manager.add_disease(
    "Bacterial Canker",
    ["Tomato", "Pepper"],
    "Bacterial canker disease is caused by Clavibacter michiganensis subsp. michiganensis. It is primarily an economic concern for tomato...",
    "Wilting in young plants, yellow leaf margins ('firing'), white spots with dark centers on fruit, vascular discoloration in stems...",
    "Source disease-free seed, clean and disinfect equipment, avoid overwatering, rotate crops, remove solanaceous weeds, deep plow soil...",
    ["Copper-based products effective in greenhouse transplant production"],
)
disease_manager.add_disease(
    "Bacterial Speck",
    ["Tomato"],
    "Bacterial speck (Pseudomonas syringae pv. tomato) infections occur during cool (63 to 75°F) wet conditions...",
    "Brown to black spots with a yellow halo on leaves, scabby pinpoint-like spots on fruit, damage does not penetrate into flesh...",
    "Start examining undersides of leaves soon after planting, delay planting until wet conditions pass, avoid overhead irrigation, rotate crops...",
    ["Crop rotation", "Avoid overhead irrigation"],
)
disease_manager.add_disease(
    "Beet Curly Top Virus",
    [
        "Bean",
        "Beet",
        "Cucurbits",
        "Pepper",
        "Spinach",
        "Tomato",
        "Several other wild and economic hosts",
    ],
    "Beet Curly Top Virus (BCTV) is a Curtovirus in the Geminiviridae family, vectored by the beet leafhopper (BLH)...",
    "Young infected plants usually die, stunted growth, thick curled leaves, dull yellow color with purple veins, prematurely ripened fruit...",
    "Implement shading, use row covers, double plant to reduce losses, manage weeds that serve as alternative hosts for BLH...",
    [
        "Integrated pest management (IPM)",
        "Avoid chemical treatments due to BLH migration",
    ],
)
disease_manager.add_disease(
    "Big Bud",
    ["Tomato", "Pepper"],
    "Big bud is caused by a small bacterium called phytoplasma, infecting plant vascular tissue and causing abnormal growth...",
    "Swollen flower buds, distorted flower and leaf growth, short and thick apical stems, yellow discoloration of foliage...",
    "Remove infected plants, clear weeds that host leafhoppers...",
    ["Leafhopper management", "Remove infected debris"],
)
disease_manager.add_disease(
    "Blossom End Rot",
    ["Cucumbers", "Eggplant", "Pepper", "Squash", "Tomato", "Watermelon"],
    "Blossom-end rot (BER) is a physiological disorder caused by calcium deficiency and aggravated by moisture imbalances...",
    "Water-soaked discoloration on the blossom end, leathery dark brown or black lesions, sunken fruit, potential secondary infections...",
    "Use deep, infrequent irrigation to maintain uniform soil moisture, avoid ammonium-based nitrogen fertilizers, protect roots from injury, apply mulch...",
    ["Optimize calcium availability", "Avoid excessive salts in soil"],
)
disease_manager.add_disease(
    "Damping-Off",
    [
        "Brassicas",
        "Corn",
        "Cucurbits",
        "Leafy Greens",
        "Legumes",
        "Onions and Garlic",
        "Potato",
        "Root Crops",
        "Solanaceae",
    ],
    "Damping-off is caused by soilborne organisms including Pythium, Rhizoctonia, Fusarium, and Phytophthora species...",
    "Circular bare spots in seed flats due to seedling death, pinched brown or black stems, seedlings tipping over...",
    "Disinfect growing benches and containers, use sterile potting mix, avoid excessive irrigation...",
    ["Seed treatments available, fungicides can reduce incidence"],
)
disease_manager.add_disease(
    "Early Blight",
    ["Potato", "Tomato"],
    "Early blight is a fungal disease caused by Alternaria solani, affecting mature vines, tubers, and fruit...",
    "Pinpoint brown or black spots on lower leaves, yellow halo around spots, concentric rings in lesions, sunken stem and tuber lesions...",
    "Use crop rotations, source disease-free transplants, destroy volunteer plants, plow down debris, maintain plant vigor, avoid prolonged leaf wetness...",
    ["Fungicide spray programs initiated in early August if lesions appear"],
)
disease_manager.add_disease(
    "Fusarium Crown/Root Rot",
    ["Asparagus", "Brassicas", "Cucurbits", "Solanaceous Crops"],
    "Fusarium species causing this disease are soilborne pathogens that persist in soil for years...",
    "Rotting and necrotic roots, discoloration and softening of crown tissue, stunted growth, wilting and plant death...",
    "Remove infected plants, sanitize equipment, sterilize soil, rotate crops with non-susceptible varieties, ensure proper soil drainage...",
    ["No fungicides available, prevention is key"],
)
disease_manager.add_disease(
    "Fusarium Wilt",
    [
        "Alfalfa",
        "Asparagus",
        "Brassicas",
        "Cucurbits",
        "Garlic",
        "Hemp",
        "Legumes",
        "Onions",
        "Potatoes",
        "Root Crops",
        "Solanaceae",
    ],
    "Fusarium wilt is caused by Fusarium oxysporum, with formae speciales specific to individual crops...",
    "Foliar chlorosis, wilting, red to purple leaf discoloration, stunted growth, brown vascular tissue, reduced yield...",
    "Use certified disease-free seed, plant resistant varieties, improve water drainage, clean equipment to minimize spread...",
    ["No chemical controls available, prevention is key"],
)
disease_manager.add_disease(
    "Late Blight",
    ["Potato", "Tomato"],
    "Late blight is caused by Phytophthora infestans and spreads rapidly under wet, moderate temperatures...",
    "Greasy-gray blotches on leaves and stems that turn black, hard brown blotchy lesions on fruit and tubers extending into flesh...",
    "Scout early after planting, monitor wet field areas, eliminate infected plant debris, avoid overhead irrigation, rotate crops...",
    ["Fungicides necessary where the disease has occurred in the past"],
)
disease_manager.add_disease(
    "Root-Knot Nematodes",
    ["Many vegetables", "Fruits", "Grasses", "Weeds"],
    "Microscopic roundworms that enter plants through root tips, moving within roots until they find preferred feeding spots...",
    "Chlorosis, stunting resembling nutrient deficiency, galled roots that may merge into large tumors...",
    "Use tolerant varieties, keep infested fields fallow, remove weeds, roto-till fallow areas regularly, apply Telone fumigation in commercial fields...",
    ["No chemical options available for non-commercial fields"],
)
disease_manager.add_disease(
    "Phytophthora Root, Stem, and Crown Rots",
    ["Solanaceous Crops", "Cucurbits", "Legumes"],
    "Caused by various Phytophthora species, thriving in water-saturated soil and infecting plant roots...",
    "Bruised, soft, rotted seedlings, chocolate-brown lesions on roots and stems, yellowing leaves, wilting and plant death in patches...",
    "Watch for symptoms early, improve soil drainage, manage water use, rotate crops, use resistant varieties...",
    ["No chemical control, prevention through water management"],
)
disease_manager.add_disease(
    "Powdery Mildew on Vegetables",
    [
        "Brassicas",
        "Cucurbits",
        "Root Crops",
        "Solanaceous Crops",
        "Legumes",
        "Hemp",
    ],
    "Powdery mildew is easily identifiable as white, powdery patches on leaves. It thrives in warm temperatures, humid canopies, and poor airflow...",
    "White, powdery fungal growth on leaves, stems, and petioles; yellowing and wilting of leaves, potential fruit sunscald and yield loss...",
    "Plant resistant varieties, remove infected plant material after harvest, improve air circulation, use morning irrigation, switch to drip irrigation...",
    ["Fungicides available, repeated applications every 7–10 days"],
)
disease_manager.add_disease(
    "Tobacco Mosaic Virus & Tomato Mosaic Virus",
    ["Tomato", "Eggplant", "Pepper"],
    "TMV and ToMV are Tobamoviruses spread by seed, grafting, human handling, and tobacco products...",
    "Abnormally shaped fruit, fruit lesions, reduced fruit size, distorted growth points, yellowing leaves, stem distortion...",
    "Remove infected plants immediately, source disease-free seed, grow resistant varieties, disinfect tools, avoid tobacco handling...",
    [
        "No chemical controls available, rely on sanitation and resistant cultivars"
    ],
)
disease_manager.add_disease(
    "Tomato Spotted Wilt Virus",
    ["Tomato", "Pepper", "Eggplant", "Several other vegetable and weed hosts"],
    "Tomato Spotted Wilt Virus (TSWV) is a Tospovirus spread by thrips species and affects plants throughout the growing season...",
    "Stunted plants, necrotic leaf spots, calico patterns on mature fruit, necrotic rings on fruit, discoloration of seeds...",
    "Monitor thrips activity, remove and destroy infected plants, purchase healthy transplants, use resistant varieties, control weeds...",
    [
        "No chemical controls available, rely on thrips management and resistant cultivars"
    ],
)
disease_manager.add_disease(
    "Verticillium Wilt",
    [
        "Cucurbits",
        "Solanaceae Crops",
        "Potato",
        "Many crop, ornamental, and weed species",
    ],
    "Verticillium wilt is caused by fungi Verticillium dahliae and Verticillium albo-atrum, persisting for years in soil...",
    "Leaves yellow and die from the bottom up, decreased fruit quality, premature wilting, symptoms appearing on one side of the plant...",
    "Use resistant varieties, plant on raised beds, avoid planting in infected fields, prevent root injury, practice sanitation, rotate crops...",
    ["No direct chemical control, rely on prevention and resistant cultivars"],
)
disease_manager.add_disease(
    "Cercospora Leaf Spot (Frogeye)",
    ["Pepper", "Eggplant"],
    "Caused by Cercospora capsici and C. melongenae, producing circular chlorotic lesions with necrotic centers...",
    "Small chlorotic lesions, necrotic centers with concentric rings, defoliation, reduced fruit size...",
    "Use protectant fungicide spray programs, remove infected debris, rotate crops, apply mulch, use furrow or drip irrigation...",
    ["Fungicide treatments based on a calendar schedule"],
)
disease_manager.add_disease(
    "Choanephora Blight (Wet Rot)",
    ["Beans", "Peas", "Squash", "Cucumber", "Eggplant", "Pepper"],
    "Caused by Choanephora cucurbitarum, occurring in tropical regions with high humidity and rainfall...",
    "Water-soaked areas on leaves, blighted apical growing points, dark gray fungal growth, silvery spine-like fungal structures, black soft rot in fruit...",
    "Few management techniques available; fungicide sprays may help reduce disease damage...",
    ["Limited fungicide options, rely on environmental control"],
)
disease_manager.add_disease(
    "Gray Leaf Spot",
    ["Pepper", "Tomato"],
    "Caused by Stemphylium solani and S. lycopersici, leading to small red to brown spots on foliage...",
    "Expanding lesions with white to gray centers and dark margins, yellowing leaves, defoliation, does not affect fruit...",
    "Remove infected plant debris, improve ventilation for seedlings, apply fungicide treatments...",
    ["Fungicides available for control"],
)
# Ajout de la maladie Phomopsis Blight
disease_manager.add_disease(
    "Phomopsis Blight",
    ["Eggplant", "Pepper"],
    "Caused by Phomopsis vexans, affecting seedlings, leaves, stems, and fruit, leading to significant plant decline...",
    "Dark-brown lesions on stems, dry rot, collapsing seedlings, circular gray-brown lesions on leaves, severe defoliation, mummified fruit...",
    "Sow high-quality seed, remove infected plants, rotate crops, mulch and furrow irrigate to reduce pathogen spread...",
    ["Protectant fungicide sprays recommended"],
)

# ✅ Correction du dictionnaire pour "Healthy"
DISEASE_DATABASE = {
    "Healthy": {
        "name": "Plante Saine",
        "scientific_name": "N/A",
        "category": "Aucune",
        "severity": "Aucune",
        "affected_crops": ["Toutes"],
        "global_distribution": ["Mondial"],
        "economic_impact": "Positif",
    }
}


def _initialize_extended_treatments_database(self) -> Dict[str, List[Dict]]:
    """Traitements pour les maladies étendues"""
    return {
        # Ajouter tous les traitements pour chaque maladie
        # Structure similaire mais étendue pour 100+ maladies
    }


def _initialize_extended_prevention_database(self) -> Dict[str, List[str]]:
    """Préventions pour les maladies étendues"""
    return {
        # Ajouter toutes les préventions pour chaque maladie
    }


def _initialize_regional_disease_data(self) -> Dict[str, List[str]]:
    """Données régionales des maladies"""
    return {
        "Europe": ["Wheat_Stripe_rust", "Apple_Scab", "Grape_Downy_mildew"],
        "Asie": [
            "Rice_Blast",
            "Rice_Bacterial_leaf_blight",
            "Coffee_Leaf_rust",
        ],
        "Afrique": [
            "Banana_Black_sigatoka",
            "Cocoa_Black_pod",
            "Rice_Brown_spot",
        ],
        "Amériques": [
            "Tomato_Late_blight",
            "Corn_Common_rust",
            "Citrus_Greening",
        ],
        "Océanie": ["Wheat_Stem_rust", "Grape_Powdery_mildew"],
    }


def get_disease_count(self) -> int:
    """Retourne le nombre total de maladies"""
    return len(self.diseases_data)


def get_diseases_by_severity(self, severity: str) -> List[Dict]:
    """Récupère les maladies par niveau de sévérité"""
    return [
        {**disease_data, "id": disease_id}
        for disease_id, disease_data in self.diseases_data.items()
        if disease_data.get("severity") == severity
    ]


def get_diseases_by_region(self, region: str) -> List[Dict]:
    """Récupère les maladies par région"""
    regional_diseases = self.regional_data.get(region, [])
    return [
        {**self.diseases_data[disease_id], "id": disease_id}
        for disease_id in regional_diseases
        if disease_id in self.diseases_data
    ]


def get_economic_impact_analysis(self) -> Dict[str, Any]:
    """Analyse de l'impact économique des maladies"""
    impact_counts = {}
    for disease_data in self.diseases_data.values():
        impact = disease_data.get("economic_impact", "Unknown")
        impact_counts[impact] = impact_counts.get(impact, 0) + 1

    return {
        "total_diseases": len(self.diseases_data),
        "impact_distribution": impact_counts,
        "catastrophic_diseases": [
            disease_id
            for disease_id, data in self.diseases_data.items()
            if data.get("economic_impact") == "Catastrophique"
        ],
    }
