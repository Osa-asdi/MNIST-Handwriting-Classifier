/**
 * MNISTLens - Frontend Controller
 * Handles drawing canvas, file uploads, sample presets, model switching,
 * and asynchronous inference communications with the Flask backend.
 */

document.addEventListener('DOMContentLoaded', () => {
    // -------------------------------------------------------------------------
    // Elements Cache
    // -------------------------------------------------------------------------
    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d');
    const canvasHint = document.getElementById('canvasHint');
    const brushSizeSlider = document.getElementById('brushSize');
    const brushSizeVal = document.getElementById('brushSizeVal');
    const clearCanvasBtn = document.getElementById('clearCanvasBtn');
    const predictCanvasBtn = document.getElementById('predictCanvasBtn');
    const realtimeToggle = document.getElementById('realtimeToggle');

    // Preview 28x28
    const previewCanvas28 = document.getElementById('previewCanvas28');
    const previewCtx28 = previewCanvas28.getContext('2d');

    // Upload Elements
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const dropzoneContent = document.getElementById('dropzoneContent');
    const uploadPreviewContainer = document.getElementById('uploadPreviewContainer');
    const uploadPreviewImg = document.getElementById('uploadPreviewImg');
    const removeUploadBtn = document.getElementById('removeUploadBtn');
    const predictUploadBtn = document.getElementById('predictUploadBtn');

    // Presets
    const presetsGrid = document.getElementById('presetsGrid');

    // Results Elements
    const predictionDigit = document.getElementById('predictionDigit');
    const confidenceBadge = document.getElementById('confidenceBadge');
    const confidenceText = document.getElementById('confidenceText');
    const latencyBadge = document.getElementById('latencyBadge');
    const latencyText = document.getElementById('latencyText');
    const resModelName = document.getElementById('resModelName');
    const resModelAcc = document.getElementById('resModelAcc');
    const resModelType = document.getElementById('resModelType');
    const modelSelector = document.getElementById('modelSelector');

    // Modal
    const openApiDocs = document.getElementById('openApiDocs');
    const closeApiDocs = document.getElementById('closeApiDocs');
    const apiDocsModal = document.getElementById('apiDocsModal');

    // Theme & URL Elements
    const themeToggle = document.getElementById('themeToggle');
    const imageUrlInput = document.getElementById('imageUrlInput');
    const fetchUrlBtn = document.getElementById('fetchUrlBtn');
    const urlPreviewCard = document.getElementById('urlPreviewCard');
    const urlPreviewImg = document.getElementById('urlPreviewImg');
    const sampleUrlChips = document.querySelectorAll('.sample-url-chip');

    // State Variables
    let isDrawing = false;
    let hasDrawn = false;
    let brushSize = parseInt(brushSizeSlider.value, 10);
    let debounceTimer = null;
    let uploadedFile = null;

    // -------------------------------------------------------------------------
    // Theme Switcher (Light / Dark Mode)
    // -------------------------------------------------------------------------
    const savedTheme = localStorage.getItem('mnist_theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);

    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('mnist_theme', newTheme);
            showNotification(`Switched to ${newTheme} mode`, 'info');
        });
    }

    // -------------------------------------------------------------------------
    // Canvas Setup
    // -------------------------------------------------------------------------
    function initCanvas() {
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = brushSize;
    }
    initCanvas();

    function clearCanvas() {
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        hasDrawn = false;
        canvasHint.style.opacity = '1';
        resetResults();
        update28Preview();
    }

    // Coordinates helper
    function getCanvasCoordinates(e) {
        const rect = canvas.getBoundingClientRect();
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        return {
            x: (clientX - rect.left) * scaleX,
            y: (clientY - rect.top) * scaleY
        };
    }

    function startDrawing(e) {
        e.preventDefault();
        isDrawing = true;
        canvas.parentElement.classList.add('drawing');
        canvasHint.style.opacity = '0';
        const { x, y } = getCanvasCoordinates(e);
        ctx.beginPath();
        ctx.moveTo(x, y);
        // Draw initial dot
        ctx.arc(x, y, brushSize / 2, 0, Math.PI * 2);
        ctx.fillStyle = '#ffffff';
        ctx.fill();
        ctx.beginPath();
        ctx.moveTo(x, y);
    }

    function draw(e) {
        if (!isDrawing) return;
        e.preventDefault();
        hasDrawn = true;
        const { x, y } = getCanvasCoordinates(e);
        ctx.lineWidth = brushSize;
        ctx.strokeStyle = '#ffffff';
        ctx.lineTo(x, y);
        ctx.stroke();

        update28Preview();

        if (realtimeToggle.checked) {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => {
                predictCanvas();
            }, 250);
        }
    }

    function stopDrawing() {
        if (!isDrawing) return;
        isDrawing = false;
        canvas.parentElement.classList.remove('drawing');
        ctx.closePath();

        update28Preview();

        if (realtimeToggle.checked && hasDrawn) {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => {
                predictCanvas();
            }, 200);
        }
    }

    // Mouse Listeners
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    window.addEventListener('mouseup', stopDrawing);

    // Touch Listeners (Mobile / Tablet friendly)
    canvas.addEventListener('touchstart', startDrawing, { passive: false });
    canvas.addEventListener('touchmove', draw, { passive: false });
    window.addEventListener('touchend', stopDrawing);

    // Brush slider
    brushSizeSlider.addEventListener('input', (e) => {
        brushSize = parseInt(e.target.value, 10);
        brushSizeVal.textContent = `${brushSize}px`;
    });

    clearCanvasBtn.addEventListener('click', clearCanvas);
    predictCanvasBtn.addEventListener('click', () => {
        if (!hasDrawn) {
            showNotification('Please draw a digit first!', 'warning');
            return;
        }
        predictCanvas();
    });

    // -------------------------------------------------------------------------
    // 28x28 Preprocessing Preview
    // -------------------------------------------------------------------------
    function update28Preview() {
        // Draw scaled down 28x28 image to show user what the ML pipeline sees
        previewCtx28.drawImage(canvas, 0, 0, 28, 28);
    }

    // -------------------------------------------------------------------------
    // Tabs Navigation
    // -------------------------------------------------------------------------
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');

    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            tabButtons.forEach(b => b.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            document.getElementById(targetTab).classList.add('active');
        });
    });

    // -------------------------------------------------------------------------
    // Upload Tab Handling
    // -------------------------------------------------------------------------
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        });
    });

    dropzone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            handleSelectedFile(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    function handleSelectedFile(file) {
        if (!file.type.startsWith('image/')) {
            showNotification('Please upload an image file (PNG, JPG, or WEBP).', 'error');
            return;
        }
        uploadedFile = file;
        const reader = new FileReader();
        reader.onload = (event) => {
            uploadPreviewImg.src = event.target.result;
            dropzoneContent.style.display = 'none';
            uploadPreviewContainer.style.display = 'block';
            predictUploadBtn.disabled = false;

            // Also draw to main canvas to inspect
            const img = new Image();
            img.onload = () => {
                ctx.fillStyle = '#000000';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                hasDrawn = true;
                canvasHint.style.opacity = '0';
                update28Preview();
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
    }

    removeUploadBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        uploadedFile = null;
        fileInput.value = '';
        dropzoneContent.style.display = 'flex';
        uploadPreviewContainer.style.display = 'none';
        predictUploadBtn.disabled = true;
    });

    predictUploadBtn.addEventListener('click', () => {
        if (!uploadedFile) return;
        predictUpload();
    });

    // -------------------------------------------------------------------------
    // Preset Digits Samples
    // -------------------------------------------------------------------------
    // Procedurally generated realistic handwritten digit templates for 0-9
    const presetTemplates = [
        // 0: smooth oval loop
        (c) => { c.ellipse(140, 140, 52, 75, 0, 0, Math.PI * 2); },
        // 1: vertical downstroke with top hook & base
        (c) => { c.moveTo(120, 85); c.lineTo(145, 60); c.lineTo(145, 220); c.moveTo(110, 220); c.lineTo(180, 220); },
        // 2: rounded top arc, diagonal stroke down-left, horizontal base
        (c) => { c.arc(140, 95, 45, Math.PI, 0); c.lineTo(95, 215); c.lineTo(190, 215); },
        // 3: two rounded loops stacked
        (c) => { c.arc(140, 95, 40, -Math.PI / 2, Math.PI / 2.2); c.arc(140, 175, 45, -Math.PI / 2.2, Math.PI / 2); },
        // 4: vertical down, horizontal right, long vertical bar
        (c) => { c.moveTo(175, 55); c.lineTo(95, 165); c.lineTo(195, 165); c.moveTo(175, 75); c.lineTo(175, 225); },
        // 5: top bar, down, rounded belly loop
        (c) => { c.moveTo(185, 65); c.lineTo(105, 65); c.lineTo(100, 130); c.arc(138, 168, 48, -Math.PI / 1.5, Math.PI / 1.8); },
        // 6: bottom closed loop with sweeping upper neck
        (c) => { c.ellipse(140, 165, 45, 48, 0, 0, Math.PI * 2); c.moveTo(98, 150); c.quadraticCurveTo(105, 65, 165, 60); },
        // 7: top horizontal bar and diagonal down
        (c) => { c.moveTo(90, 70); c.lineTo(190, 70); c.lineTo(125, 225); },
        // 8: figure eight double loops
        (c) => { c.ellipse(140, 95, 36, 36, 0, 0, Math.PI * 2); c.ellipse(140, 168, 46, 46, 0, 0, Math.PI * 2); },
        // 9: top circle loop with descending vertical right stroke
        (c) => { c.ellipse(140, 105, 42, 42, 0, 0, Math.PI * 2); c.moveTo(182, 105); c.lineTo(182, 225); }
    ];

    function initPresets() {
        presetsGrid.innerHTML = '';
        presetTemplates.forEach((drawFn, digit) => {
            const chip = document.createElement('div');
            chip.className = 'preset-chip';
            chip.title = `Load digit ${digit}`;

            const miniCanvas = document.createElement('canvas');
            miniCanvas.width = 280;
            miniCanvas.height = 280;
            miniCanvas.className = 'preset-canvas';
            const mctx = miniCanvas.getContext('2d');
            mctx.fillStyle = '#000000';
            mctx.fillRect(0, 0, 280, 280);
            mctx.lineWidth = 26;
            mctx.lineCap = 'round';
            mctx.lineJoin = 'round';
            mctx.strokeStyle = '#ffffff';
            mctx.beginPath();
            drawFn(mctx);
            mctx.stroke();

            const label = document.createElement('span');
            label.className = 'preset-label';
            label.textContent = `Digit ${digit}`;

            chip.appendChild(miniCanvas);
            chip.appendChild(label);

            chip.addEventListener('click', () => {
                // Copy preset to main canvas
                ctx.fillStyle = '#000000';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(miniCanvas, 0, 0);
                hasDrawn = true;
                canvasHint.style.opacity = '0';
                update28Preview();

                // Switch to canvas tab so user sees it
                document.getElementById('tabBtnCanvas').click();

                // Run prediction immediately
                predictCanvas();
            });

            presetsGrid.appendChild(chip);
        });
    }
    initPresets();

    // -------------------------------------------------------------------------
    // API Inference Calls
    // -------------------------------------------------------------------------
    async function predictCanvas() {
        const dataUrl = canvas.toDataURL('image/png');
        setLoadingState(true);

        try {
            const response = await fetch('/api/predict/canvas', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: dataUrl })
            });

            const data = await response.json();
            if (response.ok && data.success) {
                renderResults(data);
            } else {
                showNotification(data.error || 'Failed to predict digit', 'error');
            }
        } catch (err) {
            console.error('API Error:', err);
            showNotification('Server communication error.', 'error');
        } finally {
            setLoadingState(false);
        }
    }

    async function predictUpload() {
        if (!uploadedFile) return;
        setLoadingState(true);

        const formData = new FormData();
        formData.append('file', uploadedFile);

        try {
            const response = await fetch('/api/predict/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            if (response.ok && data.success) {
                renderResults(data);
            } else {
                showNotification(data.error || 'Failed to predict uploaded image', 'error');
            }
        } catch (err) {
            console.error('Upload API Error:', err);
            showNotification('Error classifying uploaded image.', 'error');
        } finally {
            setLoadingState(false);
        }
    }

    // -------------------------------------------------------------------------
    // Image URL Handler
    // -------------------------------------------------------------------------
    if (sampleUrlChips) {
        sampleUrlChips.forEach(chip => {
            chip.addEventListener('click', () => {
                const url = chip.getAttribute('data-url');
                if (imageUrlInput) imageUrlInput.value = url;
                predictFromUrl(url);
            });
        });
    }

    if (fetchUrlBtn) {
        fetchUrlBtn.addEventListener('click', () => {
            const url = imageUrlInput ? imageUrlInput.value.trim() : '';
            if (!url) {
                showNotification('Please enter an image URL.', 'warning');
                return;
            }
            predictFromUrl(url);
        });
    }

    async function predictFromUrl(url) {
        setLoadingState(true);
        if (fetchUrlBtn) {
            fetchUrlBtn.disabled = true;
            fetchUrlBtn.innerHTML = '<span>Fetching...</span>';
        }

        try {
            const response = await fetch('/api/predict/url', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url: url })
            });

            const data = await response.json();
            if (response.ok && data.success) {
                // Show preview card
                if (urlPreviewCard && urlPreviewImg) {
                    urlPreviewImg.src = data.image_preview;
                    urlPreviewCard.style.display = 'flex';
                }

                // Draw to canvas & 28x28 preview
                const img = new Image();
                img.onload = () => {
                    ctx.fillStyle = '#000000';
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                    hasDrawn = true;
                    canvasHint.style.opacity = '0';
                    update28Preview();
                };
                img.src = data.image_preview;

                renderResults(data);
                showNotification('Image fetched and classified successfully!', 'info');
            } else {
                showNotification(data.error || 'Failed to fetch image from URL', 'error');
            }
        } catch (err) {
            console.error('URL Fetch Error:', err);
            showNotification('Failed to download image from the provided URL.', 'error');
        } finally {
            setLoadingState(false);
            if (fetchUrlBtn) {
                fetchUrlBtn.disabled = false;
                fetchUrlBtn.innerHTML = `
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="23 4 23 10 17 10"></polyline>
                        <polyline points="1 20 1 14 7 14"></polyline>
                        <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path>
                    </svg>
                    <span>Fetch & Predict</span>
                `;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Model Switching
    // -------------------------------------------------------------------------
    modelSelector.addEventListener('change', async (e) => {
        const modelKey = e.target.value;
        try {
            const res = await fetch('/api/models/select', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_key: modelKey })
            });
            const data = await res.json();
            if (res.ok && data.success) {
                const info = data.model_info;
                resModelName.textContent = info.name;
                resModelAcc.textContent = info.test_accuracy;
                resModelType.textContent = info.type;
                showNotification(`Switched to ${info.name}`, 'info');

                // Re-predict if something is drawn
                if (hasDrawn) {
                    predictCanvas();
                }
            } else {
                showNotification(data.error || 'Failed to switch model', 'error');
            }
        } catch (err) {
            console.error('Model switch error:', err);
        }
    });

    // -------------------------------------------------------------------------
    // Render Results to UI
    // -------------------------------------------------------------------------
    function renderResults(data) {
        const { prediction, confidence, probabilities, latency_ms, model_info } = data;

        // Hero Digit
        predictionDigit.textContent = prediction;
        predictionDigit.classList.remove('pulse');
        void predictionDigit.offsetWidth; // Trigger reflow for animation
        predictionDigit.classList.add('pulse');

        // Confidence
        confidenceBadge.style.display = 'inline-block';
        confidenceText.textContent = `${confidence.toFixed(1)}% Confidence`;

        // Latency
        latencyText.textContent = `${latency_ms} ms`;
        latencyBadge.style.borderColor = 'var(--accent-cyan)';

        // Model Info
        if (model_info) {
            resModelName.textContent = model_info.name;
            resModelAcc.textContent = model_info.test_accuracy;
            resModelType.textContent = model_info.type;
        }

        // Update 28x28 Center-of-Mass Matrix Preview
        if (data.processed_preview && previewCtx28) {
            const pImg = new Image();
            pImg.onload = () => {
                previewCtx28.clearRect(0, 0, 28, 28);
                previewCtx28.drawImage(pImg, 0, 0, 28, 28);
            };
            pImg.src = data.processed_preview;
        }

        // Probabilities Breakdown
        for (let i = 0; i < 10; i++) {
            const prob = probabilities[i] !== undefined ? probabilities[i] : 0.0;
            const bar = document.getElementById(`probBar-${i}`);
            const val = document.getElementById(`probVal-${i}`);
            const row = document.getElementById(`probRow-${i}`);

            if (bar) bar.style.width = `${prob}%`;
            if (val) val.textContent = `${prob.toFixed(1)}%`;

            if (row) {
                if (i === prediction) {
                    row.classList.add('winner');
                } else {
                    row.classList.remove('winner');
                }
            }
        }
    }

    function resetResults() {
        predictionDigit.textContent = '?';
        confidenceText.textContent = 'Draw or select a digit';
        latencyText.textContent = 'Ready';

        if (previewCtx28) {
            previewCtx28.fillStyle = '#000000';
            previewCtx28.fillRect(0, 0, 28, 28);
        }

        for (let i = 0; i < 10; i++) {
            const bar = document.getElementById(`probBar-${i}`);
            const val = document.getElementById(`probVal-${i}`);
            const row = document.getElementById(`probRow-${i}`);
            if (bar) bar.style.width = '0%';
            if (val) val.textContent = '0.0%';
            if (row) row.classList.remove('winner');
        }
    }

    function setLoadingState(loading) {
        if (loading) {
            latencyText.textContent = 'Thinking...';
        }
    }

    // -------------------------------------------------------------------------
    // API Modal Handlers
    // -------------------------------------------------------------------------
    openApiDocs.addEventListener('click', (e) => {
        e.preventDefault();
        apiDocsModal.classList.add('active');
    });

    closeApiDocs.addEventListener('click', (e) => {
        e.preventDefault();
        apiDocsModal.classList.remove('active');
    });

    apiDocsModal.addEventListener('click', (e) => {
        if (e.target === apiDocsModal) {
            apiDocsModal.classList.remove('active');
        }
    });

    // -------------------------------------------------------------------------
    // Toast Notification Utility
    // -------------------------------------------------------------------------
    function showNotification(msg, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = msg;
        toast.style.position = 'fixed';
        toast.style.bottom = '24px';
        toast.style.right = '24px';
        toast.style.padding = '10px 18px';
        toast.style.borderRadius = '8px';
        toast.style.background = type === 'error' ? '#ef4444' : type === 'warning' ? '#f59e0b' : '#6366f1';
        toast.style.color = '#ffffff';
        toast.style.fontSize = '0.875rem';
        toast.style.fontWeight = '600';
        toast.style.boxShadow = '0 8px 20px rgba(0,0,0,0.4)';
        toast.style.zIndex = '9999';
        toast.style.transition = 'opacity 0.3s ease';

        document.body.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => toast.remove(), 300);
        }, 2800);
    }
});
