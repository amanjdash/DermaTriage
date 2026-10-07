export const LESION_DATA = {
  akiec: {
    code: 'akiec',
    name: 'Actinic Keratoses and Intraepithelial Carcinoma',
    friendlyName: 'Actinic Keratosis (Pre-cancerous)',
    shortLabel: 'Actinic Keratosis',
    category: 'Pre-cancerous / In Situ',
    riskLevel: 'Moderate Risk (Pre-cancerous)',
    badgeVariant: 'warning',
    overview:
      'Rough, dry, scaly patches or crusty bumps caused by chronic cumulative ultraviolet (UV) sun exposure. This class includes intraepithelial carcinoma (Bowen’s disease).',
    clinicalSignificance:
      'Pre-cancerous lesion. While not yet an invasive cancer, actinic keratoses can potentially evolve into invasive squamous cell carcinoma (SCC) if left untreated.',
    visualAppearance:
      'Typically flat to slightly elevated, pink, red, or brownish patches with a rough, sandpaper-like texture. Commonly found on sun-exposed skin such as the face, scalp, ears, forearms, and hands.',
    recommendedAction:
      'Dermatologist evaluation recommended. Standard in-office clinical treatments include cryotherapy (liquid nitrogen freezing), topical prescription creams, or photodynamic therapy.'
  },
  bcc: {
    code: 'bcc',
    name: 'Basal Cell Carcinoma',
    friendlyName: 'Basal Cell Carcinoma (Skin Cancer)',
    shortLabel: 'Basal Cell Carcinoma',
    category: 'Malignant (Common Skin Cancer)',
    riskLevel: 'High Risk (Malignant)',
    badgeVariant: 'danger',
    overview:
      'The most frequent form of skin cancer worldwide, arising from abnormal proliferation of basal cells located in the deepest layer of the epidermis.',
    clinicalSignificance:
      'Malignant skin neoplasm. Basal cell carcinoma grows locally and rarely spreads (metastasizes) to distant organs, but can cause substantial local tissue destruction if left untreated.',
    visualAppearance:
      'Often presents as a pearly, translucent bump, a pink growth with rolled elevated borders, visible tiny branching blood vessels (telangiectasias), or a persistent sore that crusts, bleeds, or will not heal.',
    recommendedAction:
      'Prompt dermatologist consultation advised for diagnostic biopsy and staging. Standard curative treatments include surgical excision, Mohs micrographic surgery, or localized curettage.'
  },
  bkl: {
    code: 'bkl',
    name: 'Benign Keratosis-like Lesions',
    friendlyName: 'Benign Keratosis (Sun / Age Spot)',
    shortLabel: 'Benign Keratosis',
    category: 'Benign (Harmless)',
    riskLevel: 'Low Risk (Benign)',
    badgeVariant: 'safe',
    overview:
      'A broad group of harmless non-cancerous skin growths, including seborrheic keratoses, solar lentigines (age/sun spots), and lichen-planus-like keratoses (LPLK).',
    clinicalSignificance:
      'Entirely benign. These lesions do not turn into melanoma or other skin cancers and carry zero risk of malignant transformation.',
    visualAppearance:
      'Usually well-demarcated round or oval patches ranging from light tan to dark brown or black. Characteristically appear "stuck on" the skin with a waxy, greasy, or finely verrucous (warty) surface.',
    recommendedAction:
      'No medical treatment is necessary. Removal (via cryotherapy or light curettage) is purely elective if a spot becomes irritated by friction from clothing or for cosmetic preferences.'
  },
  df: {
    code: 'df',
    name: 'Dermatofibroma',
    friendlyName: 'Dermatofibroma (Harmless Skin Nodule)',
    shortLabel: 'Dermatofibroma',
    category: 'Benign (Harmless)',
    riskLevel: 'Low Risk (Benign)',
    badgeVariant: 'safe',
    overview:
      'A benign, slow-growing, firm fibrous skin growth that frequently develops as a reactive response to minor skin trauma, such as an insect bite, splinter, or ingrown hair.',
    clinicalSignificance:
      'Harmless and non-cancerous. Has no potential for malignant transformation.',
    visualAppearance:
      'Small (usually 3–10 mm), firm, button-like dome or slight depression in the skin, ranging from pinkish to reddish-brown, most commonly on the lower legs. Displays the classic "pinch sign" (dimples inward when gently squeezed).',
    recommendedAction:
      'Typically left alone and requires no intervention. Medical re-evaluation is advised only if it grows rapidly, bleeds repeatedly, becomes painful, or diagnosis requires clarification.'
  },
  mel: {
    code: 'mel',
    name: 'Melanoma',
    friendlyName: 'Melanoma (Malignant Skin Cancer)',
    shortLabel: 'Melanoma',
    category: 'Malignant (High Urgency)',
    riskLevel: 'Critical / Urgent (Malignant)',
    badgeVariant: 'danger',
    overview:
      'The most serious and aggressive form of skin cancer, originating in pigment-producing melanocytes. It can arise within an existing mole or develop de novo on normal skin.',
    clinicalSignificance:
      'Highly malignant. Can invade deeply into the dermis and metastasize rapidly to regional lymph nodes and distant organs if not identified early. Early diagnosis and complete surgical excision are critical for survival.',
    visualAppearance:
      'Characteristically follows the ABCDE criteria: Asymmetry, Border irregularity (notched or jagged), Color variegation (shades of black, brown, red, white, blue), Diameter usually >6 mm, and Evolution (any change over time).',
    recommendedAction:
      'Urgent clinical evaluation required by a dermatologist or surgical oncologist. Suspected lesions require urgent excisional biopsy with appropriate histopathological margins.'
  },
  nv: {
    code: 'nv',
    name: 'Melanocytic Nevus',
    friendlyName: 'Melanocytic Nevus (Common Mole)',
    shortLabel: 'Common Mole',
    category: 'Benign (Harmless)',
    riskLevel: 'Low Risk (Benign)',
    badgeVariant: 'safe',
    overview:
      'A common, benign skin growth composed of clustered pigment-producing cells (melanocytes), widely known as a common mole or beauty mark.',
    clinicalSignificance:
      'Normal and benign. Most adults have between 10 and 40 common moles. While benign moles are harmless, routine self-monitoring helps differentiate them from evolving atypical lesions.',
    visualAppearance:
      'Typically small, symmetric, uniform tan, brown, or flesh-toned spots with smooth, distinct borders. Can be flat (junctional), slightly raised (compound), or dome-shaped (dermal).',
    recommendedAction:
      'Harmless under normal conditions. Continue routine skin self-checks following the ABCDE guidelines. Consult a clinician if any mole changes in size, shape, or color, or begins itching or bleeding.'
  },
  vasc: {
    code: 'vasc',
    name: 'Vascular Lesion',
    friendlyName: 'Vascular Lesion (Blood Vessel Growth)',
    shortLabel: 'Vascular Lesion',
    category: 'Benign (Harmless)',
    riskLevel: 'Low Risk (Benign)',
    badgeVariant: 'safe',
    overview:
      'A category of benign skin lesions formed by proliferations or malformations of superficial cutaneous blood vessels, including cherry angiomas, angiokeratomas, and pyogenic granulomas.',
    clinicalSignificance:
      'Benign vascular growths. They carry no malignancy risk.',
    visualAppearance:
      'Distinctive bright red, deep violet, or purplish spots, papules, or nodules. Some vascular growths (like pyogenic granulomas) can bleed readily upon minor friction or trauma.',
    recommendedAction:
      'Generally benign and harmless. Seek medical evaluation if the lesion bleeds repeatedly, exhibits sudden rapid enlargement, or if physical removal via electrosurgery, laser, or shaving is desired.'
  }
};

export function getLesionInfo(code) {
  const normalized = (code || '').toLowerCase().trim();
  if (LESION_DATA[normalized]) {
    return LESION_DATA[normalized];
  }
  return {
    code: code || 'unknown',
    name: code || 'Unknown Lesion',
    friendlyName: code ? `${code.toUpperCase()} Lesion` : 'Unknown Lesion',
    shortLabel: code ? code.toUpperCase() : 'Unknown',
    category: 'Unclassified Lesion',
    riskLevel: 'Clinical Assessment Required',
    badgeVariant: 'neutral',
    overview: 'No detailed encyclopedic profile is registered for this lesion class identifier.',
    clinicalSignificance: 'Unknown diagnostic category.',
    visualAppearance: 'Visual patterns vary widely.',
    recommendedAction: 'Consult a qualified dermatologist or primary care physician for comprehensive clinical dermoscopy.'
  };
}
