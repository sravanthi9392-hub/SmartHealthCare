/**
 * Smart Healthcare Interactive UI Engine
 * Features:
 * 1. Live NLP Symptom Auto-Detection from Free-Text (Auto-Toggles Symptom Chips)
 * 2. Manual Symptom Search & Toggle Customization
 * 3. Telehealth & Hospital OPD Appointment Booking Modals with Real-Time Calculations
 */

// Clinical Symptom Keywords & Synonyms Dictionary
const SYMPTOM_KEYWORD_MAP = {
    "high_fever": ["high fever", "fever", "temperature", "hot body", "burning up", "feverish", "pyrexia"],
    "mild_fever": ["mild fever", "low grade fever", "low fever", "slight temperature"],
    "chills": ["chills", "chilly", "shivering", "shiver", "feeling cold", "cold shivering"],
    "sweating": ["sweating", "sweats", "night sweats", "excessive perspiration"],
    "fatigue": ["fatigue", "tired", "tiredness", "exhaustion", "exhausted", "weakness", "weak", "drained", "no energy"],
    "lethargy": ["lethargy", "lethargic", "sluggish", "drowsy", "drowsiness"],
    "cough": ["cough", "coughing", "dry cough", "hacking cough"],
    "phlegm": ["phlegm", "mucus", "productive cough", "sputum", "wet cough", "spitting mucus"],
    "breathlessness": ["breathlessness", "shortness of breath", "breathing difficulty", "hard to breathe", "wheezing", "asthma", "gasping"],
    "chest_pain": ["chest pain", "chest tightness", "pressure in chest", "chest ache", "heaviness in chest"],
    "throat_irritation": ["sore throat", "throat pain", "throat irritation", "scratchy throat", "hurts to swallow", "throat"],
    "continuous_sneezing": ["sneezing", "sneezes", "continuous sneezing", "sneeze", "frequent sneezing"],
    "runny_nose": ["runny nose", "running nose", "cold", "flu", "watery nose", "rhinorrhea"],
    "congestion": ["blocked nose", "nasal congestion", "stuffy nose", "congestion"],
    "sinus_pressure": ["sinus", "sinus pressure", "facial pressure", "sinus headache", "forehead pressure"],
    "loss_of_smell": ["loss of smell", "no smell", "loss of taste", "cannot taste", "anosmia"],
    "stomach_pain": ["stomach pain", "stomach ache", "tummy ache", "pain in stomach", "belly ache"],
    "abdominal_pain": ["abdominal pain", "abdominal cramps", "cramps in stomach", "lower belly pain"],
    "acidity": ["acidity", "acid reflux", "heartburn", "gas", "sour burps", "gerd", "burning stomach"],
    "vomiting": ["vomiting", "vomit", "throwing up", "puking", "puke", "emesis"],
    "nausea": ["nausea", "nauseous", "queasy", "feel sick", "urge to vomit"],
    "indigestion": ["indigestion", "bloating", "bloated", "fullness", "upset stomach"],
    "diarrhoea": ["diarrhoea", "diarrhea", "loose motion", "watery stool", "frequent stools", "motions"],
    "constipation": ["constipation", "constipated", "hard stool"],
    "loss_of_appetite": ["loss of appetite", "no appetite", "not hungry", "cannot eat", "no hunger"],
    "headache": ["headache", "head ache", "head pain", "head pressure"],
    "migraine": ["migraine", "throbbing headache", "one side head pain", "aura"],
    "dizziness": ["dizziness", "dizzy", "giddy", "giddiness", "lightheaded", "vertigo"],
    "blurred_vision": ["blurred vision", "blurry vision", "cannot see clearly", "distorted vision"],
    "anxiety": ["anxiety", "anxious", "panic", "nervous", "nervousness", "stress"],
    "insomnia": ["insomnia", "sleepless", "cannot sleep", "trouble sleeping", "wake up at night"],
    "restlessness": ["restlessness", "restless", "agitated", "uneasy"],
    "fast_heart_rate": ["fast heart rate", "palpitations", "racing heart", "rapid pulse", "heart racing"],
    "joint_pain": ["joint pain", "knee pain", "aching joints", "stiff joints", "arthritis"],
    "swelling_joints": ["swollen joints", "joint swelling", "swollen knees"],
    "muscle_pain": ["muscle pain", "body ache", "body pain", "aching muscles", "myalgia"],
    "back_pain": ["back pain", "backache", "lower back pain", "spine pain"],
    "skin_rash": ["skin rash", "rash", "red spots", "skin eruptions", "red bumps"],
    "itching": ["itching", "itchy skin", "itch", "scratching", "pruritus"],
    "burning_micturition": ["burning urine", "burning sensation during urination", "pain while peeing", "uti", "burning pee"],
    "frequent_urination": ["frequent urination", "peeing often", "urinating frequently", "sugar level", "high sugar"],
    "increased_thirst": ["increased thirst", "feeling very thirsty", "excessive thirst", "dry mouth"],
    "yellowish_skin": ["yellow skin", "yellow eyes", "jaundice", "yellowish skin"],
    "dark_urine": ["dark urine", "yellow urine", "brown urine"]
};

document.addEventListener('DOMContentLoaded', function () {
    initSmartSymptomForm();
    initModalHandlers();
});

// ============================================================================
// Intelligent Live Auto-Detection & Symptom Toggler
// ============================================================================
function initSmartSymptomForm() {
    const textInput = document.getElementById('symptomsTextarea');
    const symptomTags = document.querySelectorAll('.symptom-tag');
    const selectedContainer = document.getElementById('selectedTagsContainer');
    const hiddenInput = document.getElementById('selectedSymptomsInput');
    const autoDetectAlert = document.getElementById('autoDetectAlert');
    const searchInput = document.getElementById('symptomSearch');

    if (!textInput || !symptomTags.length) return;

    let autoDetectedList = [];
    let manuallyToggledList = [];

    // 1. Live NLP Symptom Detection as the user types
    textInput.addEventListener('input', function () {
        const text = this.value.toLowerCase();
        autoDetectedList = [];

        // Check each symptom's keywords in the typed text
        for (const [symptomKey, keywords] of Object.entries(SYMPTOM_KEYWORD_MAP)) {
            for (const kw of keywords) {
                // Word boundary / substring matching
                if (text.includes(kw)) {
                    if (!autoDetectedList.includes(symptomKey)) {
                        autoDetectedList.push(symptomKey);
                    }
                    break;
                }
            }
        }

        syncSymptomTags();
    });

    // 2. Manual click toggle on badges
    symptomTags.forEach(tag => {
        tag.addEventListener('click', function () {
            const val = this.getAttribute('data-value');

            if (manuallyToggledList.includes(val)) {
                // Remove from manual
                manuallyToggledList = manuallyToggledList.filter(item => item !== val);
            } else if (autoDetectedList.includes(val)) {
                // If it was auto-detected, user clicking it acts as a manual dismiss
                autoDetectedList = autoDetectedList.filter(item => item !== val);
            } else {
                // Add to manual
                manuallyToggledList.push(val);
            }

            syncSymptomTags();
        });
    });

    // Synchronize UI active states
    function syncSymptomTags() {
        const combined = Array.from(new Set([...autoDetectedList, ...manuallyToggledList]));

        // Update active class on all tags
        symptomTags.forEach(tag => {
            const val = tag.getAttribute('data-value');
            if (combined.includes(val)) {
                tag.classList.add('selected');
                tag.innerHTML = `<i class="fa-solid fa-check"></i> ${tag.getAttribute('data-label') || tag.innerText}`;
            } else {
                tag.classList.remove('selected');
                tag.innerHTML = `+ ${tag.getAttribute('data-label') || tag.innerText}`;
            }
        });

        // Update hidden form input
        if (hiddenInput) {
            hiddenInput.value = combined.join(',');
        }

        // Update Auto-Detection Alert Banner
        if (autoDetectAlert) {
            if (autoDetectedList.length > 0) {
                autoDetectAlert.style.display = 'flex';
                const names = autoDetectedList.map(val => {
                    const el = document.querySelector(`.symptom-tag[data-value="${val}"]`);
                    return el ? (el.getAttribute('data-label') || val) : val;
                });
                document.getElementById('autoDetectCount').innerText = autoDetectedList.length;
                document.getElementById('autoDetectNames').innerText = names.join(', ');
            } else {
                autoDetectAlert.style.display = 'none';
            }
        }

        // Update count badge
        const countBadge = document.getElementById('selectedCountBadge');
        if (countBadge) {
            countBadge.innerText = `${combined.length} Selected`;
        }

        // Update Selected Chips display
        if (selectedContainer) {
            if (combined.length === 0) {
                selectedContainer.innerHTML = '<span style="color:#94a3b8; font-size:0.85rem; font-style:italic;">Type your symptoms above to auto-detect, or click any symptom badge below.</span>';
            } else {
                selectedContainer.innerHTML = combined.map(val => {
                    const el = document.querySelector(`.symptom-tag[data-value="${val}"]`);
                    const name = el ? (el.getAttribute('data-label') || el.innerText) : val;
                    const isAuto = autoDetectedList.includes(val);
                    return `<span class="badge ${isAuto ? 'badge-primary' : 'badge-success'}" style="font-size:0.825rem; padding:0.35rem 0.75rem; margin:2px;">
                        ${isAuto ? '<i class="fa-solid fa-wand-magic-sparkles"></i>' : '<i class="fa-solid fa-check"></i>'} ${name}
                        <b style="cursor:pointer; margin-left:6px;" onclick="removeSymptom('${val}')">&times;</b>
                    </span>`;
                }).join('');
            }
        }
    }

    window.removeSymptom = function(val) {
        autoDetectedList = autoDetectedList.filter(item => item !== val);
        manuallyToggledList = manuallyToggledList.filter(item => item !== val);
        syncSymptomTags();
    };

    window.clearAllSymptomTags = function() {
        autoDetectedList = [];
        manuallyToggledList = [];
        syncSymptomTags();
    };

    // Live Search Filter across all category cards
    if (searchInput) {
        searchInput.addEventListener('input', function () {
            const query = this.value.toLowerCase().trim();
            const categoryCards = document.querySelectorAll('.symptom-category-card');

            categoryCards.forEach(card => {
                const tags = card.querySelectorAll('.symptom-tag');
                let visibleCount = 0;

                tags.forEach(tag => {
                    const label = (tag.getAttribute('data-label') || tag.innerText).toLowerCase();
                    if (!query || label.includes(query)) {
                        tag.style.display = 'inline-flex';
                        visibleCount++;
                    } else {
                        tag.style.display = 'none';
                    }
                });

                if (visibleCount > 0 || !query) {
                    card.style.display = 'block';
                } else {
                    card.style.display = 'none';
                }
            });
        });
    }
}

// ============================================================================
// Enhanced Appointment Booking Modal Handlers
// ============================================================================
function initModalHandlers() {
    const modal = document.getElementById('bookingModal');
    if (!modal) return;

    window.openBookingModal = function (docName, spec, hosp, phone, type, fee, room) {
        document.getElementById('modalDocName').innerText = docName;
        document.getElementById('modalDocSpec').innerText = spec;
        document.getElementById('modalDocHosp').innerText = hosp || 'Hospital Network';

        if (document.getElementById('modalDocFee')) {
            document.getElementById('modalDocFee').innerText = fee || '₹700';
        }
        if (document.getElementById('modalDocRoom')) {
            document.getElementById('modalDocRoom').innerText = room || 'OPD Block';
        }

        document.getElementById('inputDocName').value = docName;
        document.getElementById('inputDocSpec').value = spec;
        document.getElementById('inputDocHosp').value = hosp || '';
        document.getElementById('inputDocPhone').value = phone || '';
        document.getElementById('inputAppType').value = type || 'In-Person';

        // Pre-fill tomorrow as default appointment date
        const tomorrow = new Date();
        tomorrow.setDate(tomorrow.getDate() + 1);
        const dateStr = tomorrow.toISOString().split('T')[0];
        document.getElementById('inputAppDate').value = dateStr;

        modal.classList.add('active');
    };

    window.closeBookingModal = function () {
        modal.classList.remove('active');
    };

    // Close on clicking backdrop
    modal.addEventListener('click', function (e) {
        if (e.target === modal) {
            closeBookingModal();
        }
    });
}


// Global category filter buttons handler
window.filterCategoryPills = function(catKey, btn) {
    document.querySelectorAll('.category-filter-btn').forEach(b => b.classList.remove('active'));
    if (btn) btn.classList.add('active');

    const searchInput = document.getElementById('symptomSearch');
    if (searchInput) searchInput.value = '';

    const cards = document.querySelectorAll('.symptom-category-card');
    cards.forEach(card => {
        const cardGroup = card.getAttribute('data-category-group');
        const tags = card.querySelectorAll('.symptom-tag');
        tags.forEach(t => t.style.display = 'inline-flex');

        if (catKey === 'all' || cardGroup === catKey) {
            card.style.display = 'block';
        } else {
            card.style.display = 'none';
        }
    });
};

window.clearAllSymptoms = function() {
    const textInput = document.getElementById('symptomsTextarea');
    if (textInput) textInput.value = '';
    if (window.clearAllSymptomTags) {
        window.clearAllSymptomTags();
    }
};


// Symptoms Catalog Drawer Toggle Handlers (Keeps Form Clean by Default)
window.toggleSymptomsCatalog = function() {
    const catalog = document.getElementById('symptomsCatalogDrawer');
    const textSpan = document.getElementById('catalogBtnText');
    const chevron = document.getElementById('catalogChevron');

    if (!catalog) return;

    if (catalog.style.display === 'none' || !catalog.style.display) {
        catalog.style.display = 'block';
        if (textSpan) textSpan.innerText = 'Hide Symptoms Catalog';
        if (chevron) {
            chevron.className = 'fa-solid fa-chevron-up';
        }
    } else {
        catalog.style.display = 'none';
        if (textSpan) textSpan.innerText = 'Browse Full Symptoms Catalog (50 available)';
        if (chevron) {
            chevron.className = 'fa-solid fa-chevron-down';
        }
    }
};

window.openCatalogIfClosed = function() {
    const catalog = document.getElementById('symptomsCatalogDrawer');
    const textSpan = document.getElementById('catalogBtnText');
    const chevron = document.getElementById('catalogChevron');

    if (catalog && (catalog.style.display === 'none' || !catalog.style.display)) {
        catalog.style.display = 'block';
        if (textSpan) textSpan.innerText = 'Hide Symptoms Catalog';
        if (chevron) {
            chevron.className = 'fa-solid fa-chevron-up';
        }
    }
};
