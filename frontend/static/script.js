document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Element Selections ---
    const patientForm = document.getElementById('patient-form');
    const inputScreen = document.getElementById('input-screen');
    const outputScreen = document.getElementById('output-screen');
    const historyScreen = document.getElementById('history-screen');

     // Age Selector Elements
    const ageDisplay = document.getElementById('ageDisplay');
    const hiddenAgeInput = document.getElementById('age');
    const ageDecrement5 = document.getElementById('ageDecrement5');
    const ageDecrement1 = document.getElementById('ageDecrement1');
    const ageIncrement1 = document.getElementById('ageIncrement1');
    const ageIncrement5 = document.getElementById('ageIncrement5');
    let currentAge = 40; // Default age, matching HTML

     // Lesion Location Elements
    const lesionLocationOptionsContainer = document.querySelector('.lesion-location-options');
    const hiddenLesionLocationInput = document.getElementById('lesionLocation');
    const lesionLocations = [
        "scalp", "ear", "face", "back", "trunk", "chest", 
        "upper extremity", "abdomen", "unknown", "lower extremity", 
        "genital", "neck", "hand", "foot", "acral"
    ];

    // Output screen elements
    const uploadedImageOutput = document.getElementById('uploadedImageOutput');
    const generatedReport = document.getElementById('generatedReport');
    const reportDiagnosis = document.getElementById('reportDiagnosis');
    const outputPatientName = document.getElementById('outputPatientName');
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceValue = document.getElementById('confidenceValue');

    // Input screen elements
    const imagePreview = document.getElementById('imagePreview');
    const lesionImageInput = document.getElementById('lesionImage');
    const genderButtons = document.querySelectorAll('.gender-btn');
    const hiddenGenderInput = document.getElementById('gender');

    // Header and navigation elements
    const navBack = document.querySelector('.nav-back');
    const headerTitle = document.querySelector('.header-title');
    
    // History screen elements
    const historyButton = document.getElementById('historyButton');
    const historySearchNameInput = document.getElementById('historySearchName');
    const searchHistoryBtn = document.getElementById('searchHistoryBtn');
    const historyTableBody = document.getElementById('historyTableBody');
    const historyTableContainer = document.getElementById('historyTableContainer'); // For hiding/showing table
    const noHistoryDiv = document.querySelector('.no-history');

    // Process button and loader
    const processButton = patientForm.querySelector('.process-button');
    const loader = patientForm.querySelector('.loader');

    let currentScreen = 'input'; // Tracks the currently visible screen

    // --- Screen Management ---
    function showScreen(screenName) {
        // Hide all screens
        inputScreen.style.display = 'none';
        outputScreen.style.display = 'none';
        historyScreen.style.display = 'none';
        
        navBack.style.visibility = 'hidden'; // Default to hidden for input screen
        currentScreen = screenName;

        if (screenName === 'input') {
            inputScreen.style.display = 'block';
            headerTitle.textContent = 'Input';
            patientForm.reset(); // Clear form fields
            imagePreview.style.display = 'none'; // Hide image preview
            imagePreview.src = "#"; // Reset preview src
            genderButtons.forEach(btn => btn.classList.remove('selected')); // Deselect gender buttons
            hiddenGenderInput.value = ''; // Clear hidden gender input
        } else if (screenName === 'output') {
            outputScreen.style.display = 'block';
            headerTitle.textContent = 'Output';
            navBack.style.visibility = 'visible'; // Show back button
        } else if (screenName === 'history') {
            historyScreen.style.display = 'block';
            headerTitle.textContent = 'Patient History';
            navBack.style.visibility = 'visible'; // Show back button
            loadHistory(); // Load history records when navigating to this screen
        }
    }

    // --- Event Listeners for Input Form ---
    lesionImageInput.addEventListener('change', function(event) {
        if (event.target.files && event.target.files[0]) {
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                imagePreview.style.display = 'block';
            }
            reader.readAsDataURL(event.target.files[0]);
        } else {
            imagePreview.style.display = 'none';
            imagePreview.src = "#";
        }
        if (lesionImageInput) { // Add null check for robustness
        lesionImageInput.addEventListener('change', function(event) {
            if (event.target.files && event.target.files[0]) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    if (imagePreview) {
                        imagePreview.src = e.target.result;
                        imagePreview.style.display = 'block';
                    }
                }
                reader.readAsDataURL(event.target.files[0]);
            } else {
                if (imagePreview) {
                    imagePreview.style.display = 'none';
                    imagePreview.src = "#";
                }
            }
        });
    }
    });

    genderButtons.forEach(button => {
        button.addEventListener('click', () => {
            genderButtons.forEach(btn => btn.classList.remove('selected'));
            button.classList.add('selected');
            hiddenGenderInput.value = button.dataset.value;
        });
    });

    if (ageDecrement5) ageDecrement5.addEventListener('click', () => updateAge(currentAge - 5));
    if (ageDecrement1) ageDecrement1.addEventListener('click', () => updateAge(currentAge - 1));
    if (ageIncrement1) ageIncrement1.addEventListener('click', () => updateAge(currentAge + 1));
    if (ageIncrement5) ageIncrement5.addEventListener('click', () => updateAge(currentAge + 5));

    // --- Navigation Event Listeners ---
    navBack.addEventListener('click', () => {
        // Only go back to input screen from output or history
        if (currentScreen === 'output' || currentScreen === 'history') {
            showScreen('input');
        }
        
     if (navBack) {
        navBack.addEventListener('click', () => {
            if (currentScreen === 'output' || currentScreen === 'history') {
                showScreen('input');
            }
        });
    }
    if (historyButton) historyButton.addEventListener('click', () => showScreen('history'));
    if (searchHistoryBtn && historySearchNameInput) { // Add null checks
        searchHistoryBtn.addEventListener('click', () => loadHistory(historySearchNameInput.value.trim()));
    }

    });
    
    historyButton.addEventListener('click', () => showScreen('history'));
    searchHistoryBtn.addEventListener('click', () => loadHistory(historySearchNameInput.value.trim()));

    // --- Form Submission for Prediction ---
    patientForm.addEventListener('submit', async function(event) {
        event.preventDefault();
        if (!hiddenGenderInput.value) {
            alert('Please select a gender.');
            return;
        }
        if (!lesionImageInput.files || lesionImageInput.files.length === 0) {
            alert('Please upload a lesion image.');
            return;
        }

        const formData = new FormData(patientForm);
        
        processButton.style.display = 'none'; // Hide button
        loader.style.display = 'block';      // Show loader
        processButton.disabled = true;       // Disable button

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            let result; // To store parsed JSON result

            if (!response.ok) {
                let errorMsg = `HTTP error! status: ${response.status}`;
                try {
                    const errorData = await response.json();
                    errorMsg = errorData.error || errorMsg;
                } catch (e) { /* Response might not be JSON */ }
                throw new Error(errorMsg);
            }

            result = await response.json();
            console.log("Prediction result:", result);
            console.log("Prediction result image URL from server:", result.image_url); // Debug image URL

            // Populate output screen elements with new prediction data
            populateOutputScreenWithData(result);
            showScreen('output');

        } catch (error) {
            console.error('Error during prediction:', error);
            alert('Error processing request: ' + error.message);
            // Optionally, can revert to input screen or show error on current
            // showScreen('input'); 
        } finally {
            processButton.style.display = 'block'; // Show button
            loader.style.display = 'none';       // Hide loader
            processButton.disabled = false;      // Enable button
        }
        
//////////////////////////////////
         if (patientForm) { // Add null check
        patientForm.addEventListener('submit', async function(event) {
            event.preventDefault();

            if (hiddenGenderInput && !hiddenGenderInput.value) { // Check if element exists
                alert('Please select a gender.');
                return;
            }
            // Age is implicitly validated by the updateAge function (always a number between 10-90)
            // and the hiddenAgeInput will always have a value if the elements exist.

            if (hiddenLesionLocationInput && !hiddenLesionLocationInput.value) { // Check if element exists
                alert('Please select a lesion location.');
                return;
            }
            if (lesionImageInput && (!lesionImageInput.files || lesionImageInput.files.length === 0)) { // Check if element exists
                alert('Please upload a lesion image.');
                return;
            }

            const formData = new FormData(patientForm);
            // Log formData for debugging :
            // console.log("Form Data being sent:");
            // for (let [key, value] of formData.entries()) {
            //     console.log(`${key}: ${value}`);
            // }
            
            if (processButton) processButton.style.display = 'none';
            if (loader) loader.style.display = 'block';
            if (processButton) processButton.disabled = true;

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                let result; 

                if (!response.ok) {
                    let errorMsg = `HTTP error! status: ${response.status}`;
                    try {
                        const errorData = await response.json();
                        errorMsg = errorData.error || errorMsg;
                    } catch (e) { /* Response might not be JSON */ }
                    throw new Error(errorMsg);
                }

                result = await response.json();
                console.log("Prediction result:", result);
                if(result.image_url) console.log("Prediction result image URL from server:", result.image_url);

                populateOutputScreenWithData(result);
                showScreen('output');

            } catch (error) {
                console.error('Error during prediction:', error);
                alert('Error processing request: ' + error.message);
            } finally {
                if (processButton) processButton.style.display = 'block';
                if (loader) loader.style.display = 'none';
                if (processButton) processButton.disabled = false;
            }
        });
    }/////////////////


    });

     // --- Function to Initialize Lesion Location Buttons ---
    function initializeLesionLocationButtons() {
        lesionLocations.forEach(location => {
            const button = document.createElement('button');
            button.type = 'button';
            button.classList.add('option-btn', 'lesion-location-btn');
            button.dataset.value = location;
            button.textContent = location.charAt(0).toUpperCase() + location.slice(1); // Capitalize
            
            button.addEventListener('click', () => {
                // Deselect other lesion location buttons
                lesionLocationOptionsContainer.querySelectorAll('.lesion-location-btn').forEach(btn => btn.classList.remove('selected'));
                // Select clicked button
                button.classList.add('selected');
                hiddenLesionLocationInput.value = location;
            });
            lesionLocationOptionsContainer.appendChild(button);
        });
    }

    // --- Function to Update Age Display and Hidden Input ---
    function updateAge(newAge) {
        currentAge = Math.max(1, Math.min(100, newAge)); // Clamp age between 1 and 100
        ageDisplay.textContent = currentAge;
        hiddenAgeInput.value = currentAge;
    }
    updateAge(currentAge); // Initialize with default age

    // --- Screen Management  ---
    function showScreen(screenName) {
          inputScreen.style.display = 'none';
        outputScreen.style.display = 'none';
        historyScreen.style.display = 'none';
        
        navBack.style.visibility = 'hidden';
        currentScreen = screenName;

        if (screenName === 'input') {
            inputScreen.style.display = 'block';
            headerTitle.textContent = 'Input';
            patientForm.reset();
            imagePreview.style.display = 'none';
            imagePreview.src = "#";
            
            // Reset Gender Buttons
            genderButtons.forEach(btn => btn.classList.remove('selected'));
            if (hiddenGenderInput) hiddenGenderInput.value = '';
            
            // Reset Age to default
            if (ageDisplay && hiddenAgeInput) updateAge(40); 
            
            // Reset Lesion Location Buttons
            if (lesionLocationOptionsContainer) {
                 lesionLocationOptionsContainer.querySelectorAll('.lesion-location-btn').forEach(btn => btn.classList.remove('selected'));
            }
            if (hiddenLesionLocationInput) hiddenLesionLocationInput.value = '';

        } else if (screenName === 'output') {
            outputScreen.style.display = 'block';
            headerTitle.textContent = 'Output';
            navBack.style.visibility = 'visible';
        } else if (screenName === 'history') {
            historyScreen.style.display = 'block';
            headerTitle.textContent = 'Patient History';
            navBack.style.visibility = 'visible';
            loadHistory();
        }
    }
    

    // --- Function to Populate Output Screen (Reusable) ---
    function populateOutputScreenWithData(data) {
        // data can be 'result' from prediction or 'record' from history
        const imageUrl = data.image_url; // Expecting this to be like /filename.ext
        console.log("Setting output image src to:", imageUrl); // Debug image URL
        uploadedImageOutput.src = imageUrl; 
        uploadedImageOutput.onerror = () => { // Add onerror for debugging
            console.error("Error loading image on output screen:", imageUrl);
            uploadedImageOutput.alt = "Image failed to load";
        };


        generatedReport.textContent = data.report_text;
        reportDiagnosis.textContent = data.report_diagnosis_confidence || 
                                     (data.diagnosis_full ? `Diagnosis: ${data.diagnosis_full} (Confidence: ${(data.confidence * 100).toFixed(1)}%)` : `Diagnosis: ${data.diagnosis_short} (Confidence: ${(data.confidence * 100).toFixed(1)}%)`);
        outputPatientName.textContent = data.patient_name || "Patient";
        
        const confidencePercent = (data.confidence * 100).toFixed(1);
        confidenceBar.style.width = `${confidencePercent}%`;
        confidenceValue.textContent = `${confidencePercent}%`;
        
        if (data.confidence < 0.5) confidenceBar.style.backgroundColor = '#f44336'; // Red
        else if (data.confidence < 0.75) confidenceBar.style.backgroundColor = '#ffc107'; // Yellow
        else confidenceBar.style.backgroundColor = '#4CAF50'; // Green
    }


    // --- History Management ---
    async function loadHistory(patientName = '') {
        try {
            const response = await fetch(`/history?patientName=${encodeURIComponent(patientName)}`);
            if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
            
            const records = await response.json(); // Expects an array of records
            console.log("History records received:", records);
            
            historyTableBody.innerHTML = ''; // Clear previous records
            if (records.length === 0) {
                noHistoryDiv.style.display = 'block';
                if (historyTableContainer) historyTableContainer.style.display = 'none'; // Hide table if no records
            } else {
                noHistoryDiv.style.display = 'none';
                if (historyTableContainer) historyTableContainer.style.display = 'block'; // Show table
                records.forEach(record => {
                    const tr = document.createElement('tr');
                    const recordDate = new Date(record.timestamp).toLocaleDateString();
                    // Backend now sends record.image_url as /filename.ext for history items too
                    const thumbnailUrl = record.image_url; 
                    console.log("History thumbnail URL:", thumbnailUrl); // Debug thumbnail URL

                    tr.innerHTML = `
                        <td>${recordDate}</td>
                        <td>${record.patient_name}</td>
                        <td>${record.diagnosis_short} (${(record.confidence * 100).toFixed(1)}%)</td>
                        <td><img src="${thumbnailUrl}" alt="Lesion" class="history-thumbnail" data-fullimage="${thumbnailUrl}"></td>
                        <td><button class="view-details-btn" data-record='${JSON.stringify(record)}'>Details</button></td>
                    `;
                    // Add onerror to history thumbnail images for debugging
                    const imgElement = tr.querySelector('.history-thumbnail');
                    imgElement.onerror = () => {
                        console.error("Error loading history thumbnail:", thumbnailUrl);
                        imgElement.alt = "Img fail"; // Indicate failure
                    };
                    historyTableBody.appendChild(tr);
                });
            }
        } catch (error) {
            console.error('Error loading history:', error);
            historyTableBody.innerHTML = `<tr><td colspan="5">Error loading history. Check console.</td></tr>`;
            noHistoryDiv.style.display = 'block';
            if (historyTableContainer) historyTableContainer.style.display = 'none';
        }
    }

    // Event delegation for dynamically added elements in the history table
    historyTableBody.addEventListener('click', function(event) {
        const target = event.target;
        if (target.classList.contains('history-thumbnail')) {
            const fullImageUrl = target.dataset.fullimage;
            if (fullImageUrl) showImageModal(fullImageUrl);
        }
        if (target.classList.contains('view-details-btn')) {
            try {
                const recordData = JSON.parse(target.dataset.record);
                if (recordData) {
                    populateOutputScreenWithData(recordData); // Use the reusable function
                    showScreen('output'); // Navigate to the output screen
                }
            } catch(e) {
                console.error("Error parsing record data from history button:", e);
                alert("Could not load details for this record.");
            }
        }
    });

    // --- Modal for Viewing Full Image (from history) ---
    function showImageModal(imageUrl) {
        let modal = document.getElementById('imageModal');
        if (!modal) { // Create modal if it doesn't exist
            modal = document.createElement('div');
            modal.id = 'imageModal';
            modal.classList.add('modal');
            modal.innerHTML = `<span class="close-modal">×</span><img class="modal-content" id="modalImageSrc">`;
            document.body.appendChild(modal);
            
            const closeModalButton = modal.querySelector('.close-modal');
            if (closeModalButton) {
                closeModalButton.onclick = () => modal.style.display = "none";
            }
            modal.onclick = (e) => { // Close on outside click
                if (e.target === modal) {
                    modal.style.display = "none";
                }
            };
        }
        const modalImage = modal.querySelector('#modalImageSrc');
        if (modalImage) {
            modalImage.src = imageUrl;
        }
        modal.style.display = "flex"; // Use flex for centering
    }

    // --- Initial Application Setup ---
     if (lesionLocationOptionsContainer) { // Check if element exists
        initializeLesionLocationButtons();
    }
    showScreen('input'); // Start with the input screen
});