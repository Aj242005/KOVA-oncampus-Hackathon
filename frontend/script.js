const API_BASE_URL = 'http://127.0.0.1:5000';

document.addEventListener("DOMContentLoaded", () => {
    const container = document.getElementById('ranked-candidates');

    fetch(`${API_BASE_URL}/ranked`)
        .then(response => {
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            return response.json();
        })
        .then(jsonResponse => {
            if (jsonResponse.status !== 'success') {
                container.innerHTML = `<p style="color: red;">System Failure: ${jsonResponse.message}</p>`;
                return;
            }

            container.innerHTML = ''; // Clear loading text
            const data = jsonResponse.data;

            Object.keys(data).forEach(jdId => {
                const jd = data[jdId];
                
                // Create the wrapper for the JD layout
                const jdSection = document.createElement('div');
                jdSection.className = 'jd-section';
                
                // Left column: JD Meta
                const jdHeader = `
                    <div class="jd-header">
                        <div class="jd-id">ID // ${jdId}</div>
                        <h2 class="jd-role">${jd.role}</h2>
                        <div class="jd-company">@ ${jd.company}</div>
                    </div>
                `;

                // Right column: Candidates
                const candidatesContainer = document.createElement('div');
                candidatesContainer.className = 'jd-candidates';

                jd.top_candidates.forEach(candidate => {
                    const candidateElement = document.createElement('div');
                    candidateElement.className = 'candidate';
                    
                    const skillsHtml = candidate.matched_skills.length > 0 
                        ? candidate.matched_skills.map(skill => `<span class="skill-tag">${skill}</span>`).join('')
                        : '<span class="skill-tag" style="color: #666; border-color: #333;">No direct matches</span>';

                    candidateElement.innerHTML = `
                        <div class="candidate-header">
                            <div>
                                <div class="candidate-rank">RANK 0${candidate.rank}</div>
                                <h3 class="candidate-name">${candidate.name}</h3>
                            </div>
                            <div class="candidate-score">
                                <div class="score-value">${candidate.score.toFixed(2)}</div>
                                <div class="score-label">Match Score</div>
                            </div>
                        </div>
                        <div class="skills-container">
                            ${skillsHtml}
                        </div>
                    `;
                    candidatesContainer.appendChild(candidateElement);
                });

                // Assemble the section
                jdSection.innerHTML = jdHeader;
                jdSection.appendChild(candidatesContainer);
                container.appendChild(jdSection);
            });
        })
        .catch(error => {
            console.error('Error fetching data:', error);
            container.innerHTML = `<p style="color: #ff3333;">Connection lost. Ensure the Python engine is running on ${API_BASE_URL}.</p>`;
        });
});
