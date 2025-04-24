document.getElementById('review-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const reviewInput = document.getElementById('review-input');
    const submitBtn = document.getElementById('submit-btn');
    const errorDiv = document.getElementById('error');
    const resultDiv = document.getElementById('result');
    
    errorDiv.style.display = 'none';
    resultDiv.style.display = 'none';
    submitBtn.disabled = true;
    submitBtn.textContent = 'Analyzing...';
    
    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ review: reviewInput.value })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            resultDiv.innerHTML = `
                <h3>Prediction</h3>
                <p>Sentiment: <span class="sentiment-${data.sentiment}">${data.sentiment}</span></p>
                <h4>Probabilities</h4>
                <ul>
                    <li>Negative: ${(data.probabilities.negative * 100).toFixed(2)}%</li>
                    <li>Neutral: ${(data.probabilities.neutral * 100).toFixed(2)}%</li>
                    <li>Positive: ${(data.probabilities.positive * 100).toFixed(2)}%</li>
                </ul>
            `;
            resultDiv.style.display = 'block';
        } else {
            errorDiv.textContent = data.error || 'An error occurred';
            errorDiv.style.display = 'block';
        }
    } catch (err) {
        errorDiv.textContent = 'Failed to connect to the server';
        errorDiv.style.display = 'block';
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Analyze Sentiment';
    }
});