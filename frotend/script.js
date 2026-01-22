// ===== 중복요청 취소 + 캐시(SWR) =====
let inflightController = null;             
const CACHE_TTL_MS = 5 * 60 * 1000;          
const memoryCache = new Map();

function _normalize(u) { return (u || "").trim(); }
function _cacheKey(url) { return `analyze::${_normalize(url).toLowerCase()}`; }

function _getCached(url) {
    const key = _cacheKey(url);
    const now = Date.now();

    const mem = memoryCache.get(key);
    if (mem && now - mem.t <= CACHE_TTL_MS) return mem.d;

    const raw = localStorage.getItem(key);
    if (raw) {
        try {
            const obj = JSON.parse(raw);
            if (now - obj.t <= CACHE_TTL_MS) {
                memoryCache.set(key, { t: obj.t, d: obj.d });
                return obj.d;
            } else {
                localStorage.removeItem(key);
            }
        } catch { /* ignore */ }
    }
    return null;
}
function _setCached(url, data) {
    const key = _cacheKey(url);
    const v = { t: Date.now(), d: data };
    memoryCache.set(key, v);
    try { localStorage.setItem(key, JSON.stringify(v)); } catch {}
}

// ===== analyzeUrl =====
async function analyzeUrl() {
    const urlInput = document.getElementById('urlInput');
    const resultArea = document.getElementById('resultArea');
    const resultCard = document.querySelector('.result-card'); 
    const errorMessage = document.getElementById('errorMessage');
    const urlToAnalyze = urlInput.value.trim();

    errorMessage.textContent = '';
    resultCard.className = 'result-card result-initial'; 
    resultArea.innerHTML = `
        <div class="result-initial">
            <h4>Analysis Result</h4>
            <p>Results will be displayed here.</p>
        </div>
    `;

    // URL 형식 검사 로직 
    const hangulRegex = /[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]/; 
    const numbersOnlyRegex = /^[0-9]+$/;   

    if (!urlToAnalyze || hangulRegex.test(urlToAnalyze) || numbersOnlyRegex.test(urlToAnalyze)) {
        errorMessage.textContent = '올바른 URL 형식을 입력해주세요. (예: https://example.com)';
        return;
    }

    // ===== 캐시(SWR): 캐시가 있으면 즉시 표시, 백그라운드로 최신화 =====
    const cached = _getCached(urlToAnalyze);
    if (cached) {
        resultCard.className = 'result-card';
        resultArea.innerHTML = '';

        const scorePercent = (cached.confidence_score * 100).toFixed(2);
        let riskLevelClass = 'risk-low';
        let actionGuide = '';
        let actionGuideClass = '';

        if (scorePercent > 70) {
            riskLevelClass = 'risk-high';
        } else if (scorePercent > 30) {
            riskLevelClass = 'risk-medium';
        }

        if (cached.is_malicious) {
            actionGuide = '방문을 절대 권장하지 않습니다. 즉시 브라우저 탭을 닫아주세요.';
            actionGuideClass = 'guide-malicious';
        } else {
            actionGuide = '알려진 위협이 발견되지 않았습니다. 안전하게 방문할 수 있습니다.';
            actionGuideClass = 'guide-normal';
        }

        const riskBarHtmlCached = `
            <div class="risk-bar-container">
                <div class="risk-bar-fill ${riskLevelClass}" style="width: ${scorePercent}%;"></div>
            </div>
            <p class="action-guide">${actionGuide}</p> `;

        if (cached.is_malicious) {
            resultArea.innerHTML = `
                <div class="result-malicious-text">
                    <h2>⚠️ 악성 URL</h2>
                    <p>위험도: ${scorePercent}%</p>
                    ${riskBarHtmlCached}
                </div>`;
            resultCard.classList.add('result-malicious');
        } else {
            resultArea.innerHTML = `
                <div class="result-normal-text">
                    <h2>✅ 정상 URL</h2>
                    <p>위험도: ${scorePercent}%</p>
                    ${riskBarHtmlCached}
                </div>`;
            resultCard.classList.add('result-normal');
        }
    } else {
        resultCard.classList.add('result-loading'); 
        resultArea.innerHTML = `
            <div class="spinner"></div>
            <p>Analyzing...</p>
        `;
    }

    // ===== 중복요청 취소 + 네트워크 요청 =====
    if (inflightController) inflightController.abort();
    inflightController = new AbortController();
    const signal = inflightController.signal;
    
    try {
        const response = await fetch('http://127.0.0.1:8000/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: urlToAnalyze }),
            signal, 
        });

        if (!response.ok) {
            throw new Error(`Server error! Status: ${response.status}`);
        }

        const result = await response.json();

        // 최신 결과를 캐시에 저장
        _setCached(urlToAnalyze, result);
        
        resultCard.className = 'result-card';
        resultArea.innerHTML = '';

        const scorePercent = (result.confidence_score * 100).toFixed(2);
        let riskLevelClass = 'risk-low';
        let actionGuide = '';
        let actionGuideClass = '';

        if (scorePercent > 70) {
            riskLevelClass = 'risk-high';
        } else if (scorePercent > 30) {
            riskLevelClass = 'risk-medium';
        }

        if (result.is_malicious) {
            actionGuide = '방문을 절대 권장하지 않습니다. 즉시 브라우저 탭을 닫아주세요.';
            actionGuideClass = 'guide-malicious';
        } else {
            actionGuide = '알려진 위협이 발견되지 않았습니다. 안전하게 방문할 수 있습니다.';
            actionGuideClass = 'guide-normal';
        }

        const riskBarHtml = `
            <div class="risk-bar-container">
                <div class="risk-bar-fill ${riskLevelClass}" style="width: ${scorePercent}%;"></div>
            </div>
            <p class="action-guide">${actionGuide}</p> `;

        if (result.is_malicious) {
            resultArea.innerHTML = `
                <div class="result-malicious-text">
                    <h2>⚠️ 악성 URL</h2>
                    <p>위험도: ${scorePercent}%</p>
                    ${riskBarHtml}
                </div>`;
            resultCard.classList.add('result-malicious');
        } else {
            resultArea.innerHTML = `
                <div class="result-normal-text">
                    <h2>✅ 정상 URL</h2>
                    <p>위험도: ${scorePercent}%</p>
                    ${riskBarHtml}
                </div>`;
            resultCard.classList.add('result-normal');
        }

    } catch (error) {
        if (error.name === 'AbortError') return; 
        console.error('Fetch Error:', error);
        if (!cached) {
            resultArea.innerHTML = `<div class="result-error-text">
                <h2>⚠️ 오류 발생</h2>
                <p>분석 중 오류가 발생했습니다. <br>
                문제가 계속된다면 네트워크 연결을 확인해주세요.</p></div>`;
            resultCard.classList.add('result-error'); 
        }
    } finally {
        inflightController = null;
    }
}

// 페이지 로딩 후 실행 
document.addEventListener('DOMContentLoaded', function() {
    const urlInput = document.getElementById('urlInput');
    const ctaButton = document.querySelector('.cta-button');
    const analyzeNavLink = document.getElementById('nav-analyze-link');
    const contentToFade = document.querySelector('.header-text-container');
    const resultCard = document.querySelector('.result-card');
    const resultArea = document.getElementById('resultArea');
    const errorMessage = document.getElementById('errorMessage');
    const techSection = document.querySelector('.tech-section');
    const techNavLink = document.getElementById('nav-tech-link'); 
    const benefitsSection = document.querySelector('#benefits');
    const benefitsNavLink = document.getElementById('nav-benefits-link');

    function resetResultArea() {
        if (resultCard && !resultCard.classList.contains('result-initial')) {
            resultCard.className = 'result-card result-initial';
            resultArea.innerHTML = `
                <div class="result-initial">
                    <h4>Analysis Result</h4>
                    <p>Results will be displayed here.</p>
                </div>
            `;
        }

        if (urlInput) {
            urlInput.value = '';
        }

        if (errorMessage) {
            errorMessage.textContent = '';
        }
    }

    if (urlInput) {
        urlInput.placeholder = 'Enter URL to analyze...';

        urlInput.addEventListener('keydown', function(event) {
            if (event.key === 'Enter') {
                event.preventDefault();
                analyzeUrl();
            }
        });

        urlInput.addEventListener('input', function() {
            const errorMessage = document.getElementById('errorMessage');
            if (errorMessage && errorMessage.textContent !== '') {
                errorMessage.textContent = '';
            }
        });

        urlInput.addEventListener('focus', resetResultArea);
    }

    if (ctaButton) {
        ctaButton.addEventListener('click', function(event) {
            event.preventDefault();
            resetResultArea();
            document.getElementById('detector').scrollIntoView({ behavior: 'smooth' });
        });
    }
    
    if (analyzeNavLink) {
        analyzeNavLink.addEventListener('click', function(event) {
            event.preventDefault();
            resetResultArea(); 
            document.getElementById('detector').scrollIntoView({ behavior: 'smooth' });
        });
    }

    if (contentToFade) {
        window.addEventListener('scroll', function() {
            const scrollPosition = window.scrollY;
            const fadeOutHeight = window.innerHeight * 0.8;

            if (scrollPosition < fadeOutHeight) {
                const opacity = 1 - (scrollPosition / fadeOutHeight);
                contentToFade.style.opacity = opacity;
                const blur = (scrollPosition / fadeOutHeight) * 5;
                contentToFade.style.filter = `blur(${blur}px)`;
            } else {
                contentToFade.style.opacity = 0;
                contentToFade.style.filter = 'blur(5px)';
            }
        });
    }

    if (techSection) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                } else {
                    entry.target.classList.remove('visible');
                }
            });
        }, {
            threshold: 0.1
        });
        observer.observe(techSection);
    }

    if (techNavLink && techSection) {
        techNavLink.addEventListener('click', function(event) {
            event.preventDefault();
            techSection.classList.remove('visible');
            techSection.scrollIntoView({ behavior: 'smooth' });
            setTimeout(() => {
                techSection.classList.add('visible');
            }, 300);
        });
    }

    if (benefitsSection) {
        const benefitsObserver = new IntersectionObserver((entries, observer) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('is-visible');
                } else {
                    entry.target.classList.remove('is-visible');
                }
            });
        }, {
            threshold: 0.2 
        });
        benefitsObserver.observe(benefitsSection);
    }

    if (benefitsNavLink && benefitsSection) {
        benefitsNavLink.addEventListener('click', function(event) {
            event.preventDefault();

            benefitsSection.classList.remove('is-visible');
            benefitsSection.scrollIntoView({ behavior: 'smooth' });

            setTimeout(() => {
                benefitsSection.classList.add('is-visible');
            }, 20);
        });
    }
});
