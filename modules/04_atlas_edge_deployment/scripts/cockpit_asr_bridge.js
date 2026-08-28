// Car-ASR Bridge - 语音控制桥接
(function() {
  'use strict';

  document.addEventListener('DOMContentLoaded', function() {
    const pttBtn = document.getElementById('push-to-talk');
    const voiceText = document.getElementById('voice-btn-text');
    const statusText = document.getElementById('audio-status-text');

    if (!pttBtn) return;

    let mr = null, chunks = [], recording = false, starting = false;


    async function startRecording() {
      if (recording || starting) return;
      starting = true;
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mr = new MediaRecorder(stream);
        chunks = [];
        mr.ondataavailable = e => chunks.push(e.data);
        mr.onstop = async () => {
          stream.getTracks().forEach(t => t.stop());
          await sendToASR();
        };
        mr.start();
        recording = true;
        starting = false;
        voiceText.textContent = '松开识别';
        statusText.textContent = '聆听中…';
        pttBtn.style.background = 'rgba(255,80,80,0.25)';
        pttBtn.style.color = '#fff';
      } catch(e) {
        starting = false;
        statusText.textContent = '麦克风未授权';
      }
    }

    function stopRecording() {
      if (!recording) { starting = false; return; }
      try { mr.stop(); } catch(e) {}
      recording = false;
      starting = false;
      voiceText.textContent = '按住说话';
      statusText.textContent = '识别中…';
      pttBtn.style.background = '';
      pttBtn.style.color = '';
    }

    pttBtn.addEventListener('mousedown', e => { e.preventDefault(); startRecording(); });
    pttBtn.addEventListener('mouseup', stopRecording);
    pttBtn.addEventListener('mouseleave', stopRecording);
    pttBtn.addEventListener('touchstart', e => { e.preventDefault(); startRecording(); });
    pttBtn.addEventListener('touchend', e => { e.preventDefault(); stopRecording(); });
    pttBtn.addEventListener('touchcancel', stopRecording);

    async function sendToASR() {
      if (!chunks.length) { statusText.textContent = '未检测到语音'; return; }
      const blob = new Blob(chunks, { type: 'audio/webm' });
      const form = new FormData();
      form.append('file', blob, 'rec.webm');
      try {
        const res = await fetch('/api/recognize', { method: 'POST', body: form });
        const data = await res.json();
        const ctrl = window.cockpitController;

        if (data.text) {
          statusText.textContent = '"' + data.text + '" (' + data.delay_ms + 'ms)';
          if (data.command && data.command.actions && ctrl) {
            for (const a of data.command.actions) {
              applyCommand(a, ctrl);
            }
          }
        } else {
          statusText.textContent = '未识别到语音';
        }
        setTimeout(() => { if (!recording) statusText.textContent = '就绪'; }, 3000);
      } catch(e) {
        statusText.textContent = '识别失败，请重试';
      }
    }

    function applyCommand(a, ctrl) {
      switch(a.type) {
        case 'climate':
          if (a.target === 'temperature') {
            if (a.value) ctrl.climateSetTarget(a.value);
            else if (a.delta) ctrl.climateAdjustTarget(a.delta);
          }
          if (a.target === 'fan_speed') {
            if (a.value) ctrl.climateSetFan(a.value);
            else if (a.delta) ctrl.climateAdjustFan(a.delta);
          }
          if (a.target === 'ac') {
            document.getElementById('ac-badge').textContent = a.value ? '开启' : '关闭';
            const btn = document.getElementById('ac-toggle-btn');
            if (btn) { btn.textContent = '❄️ 空调: ' + (a.value ? '开' : '关'); btn.style.background = a.value ? 'rgba(0,200,255,0.15)' : ''; }
          }
          if (a.target === 'defrost') {
            const btn = document.getElementById('defrost-toggle-btn');
            if (btn) { btn.textContent = '🌫 除雾: 开'; btn.style.background = 'rgba(0,200,255,0.15)'; }
          }

          if (a.target === 'seat_heat') {
            const btn = document.getElementById('rear-def-btn');
            if (btn && a.value !== undefined) {
              btn.textContent = '♨ 座椅加热: ' + (a.value ? '开' : '关');
              btn.style.background = a.value ? 'rgba(0,200,255,0.15)' : '';
            }
          }
          if (a.target === 'circulation') {
            const btn = document.getElementById('circ-toggle-btn');
            if (btn) { btn.textContent = '🔄 内循环: ' + (a.value ? '开' : '关'); btn.style.background = a.value ? 'rgba(0,200,255,0.15)' : ''; }
          }
          break;
                case 'window':
          const targets = a.target === 'all' ? ['fl','fr','rl','rr'] : [a.target];
          for (const t of targets) {
            const btn = document.querySelector('[data-window-toggle="'+t+'"]');
            if (btn) {
              const wantOpen = a.action === 'open';
              const isOpen = btn.textContent === '关闭';
              // Only click if state differs
              if (wantOpen !== isOpen) btn.click();
            }
          }
          break;case 'media':
          if (a.target === 'play') {
            if (a.value) ctrl.mediaPlay();
            else ctrl.mediaPause();
          }
          if (a.target === 'next') {
            ctrl.mediaNext();
          }
          if (a.target === 'previous') {
            ctrl.mediaPrevious();
          }
          if (a.target === 'volume') {
            // Volume indicator only (controller has no volume method)
            const currentVol = document.getElementById('media-status-text');
            if (currentVol && a.delta) {
              const dir = a.delta > 0 ? '🔊' : '🔉';
              currentVol.textContent = dir + ' 音量' + (a.delta > 0 ? '+' : '') + a.delta;
              setTimeout(() => {
                const track = ctrl.media.tracks[ctrl.media.currentIndex];
                if (track) currentVol.textContent = ctrl.media.isPlaying ? '播放中' : '已暂停';
              }, 2000);
            }
          }
          break;

                        case 'door':
          if (a.target === 'trunk') {
            statusText.textContent = a.action === 'open' ? '后备箱已开启' : '后备箱已关闭';
          } else if (a.target === 'lock') {
            statusText.textContent = a.value ? '车辆已上锁' : '车辆已解锁';
          } else {
            const doorToWindow = {fl:'fl', fr:'fr', rl:'rl', rr:'rr', rear:'rl'};
            const targets = a.target === 'all' ? ['fl','fr','rl','rr'] : [doorToWindow[a.target] || a.target];
            const wantClose = a.action === 'close';
            for (const t of targets) {
              const btn = document.querySelector('[data-window-toggle="'+t+'"]');
              if (!btn) continue;
              const isOpen = btn.textContent === '关闭';
              // Click to toggle only if state differs from desired state
              if (wantClose ? isOpen : !isOpen) btn.click();
            }
            statusText.textContent = wantClose ? '车门已关闭' : '车门已开启';
          }
          break;case 'nav':
          if (a.destination && ctrl) {
            ctrl.navSetDestination(a.destination);
            setTimeout(() => ctrl.navStart(), 500);
          }
          break;
      }
    }

    // Toggle buttons for climate panel
    window.toggleAC = function() {
      const btn = document.getElementById('ac-toggle-btn');
      const on = !btn.textContent.includes('开');
      btn.textContent = '❄️ 空调: ' + (on ? '开' : '关');
      btn.style.background = on ? 'rgba(0,200,255,0.15)' : '';
      document.getElementById('ac-badge').textContent = on ? '开启' : '关闭';
    };
    window.toggleDefrost = function() {
      const btn = document.getElementById('defrost-toggle-btn');
      const on = !btn.textContent.includes('开');
      btn.textContent = '🌫 除雾: ' + (on ? '开' : '关');
      btn.style.background = on ? 'rgba(0,200,255,0.15)' : '';
    };
    window.toggleCirc = function() {
      const btn = document.getElementById('circ-toggle-btn');
      const on = !btn.textContent.includes('开');
      btn.textContent = '🔄 内循环: ' + (on ? '开' : '关');
      btn.style.background = on ? 'rgba(0,200,255,0.15)' : '';
    };
    window.toggleRearDef = function() {
      const btn = document.getElementById('rear-def-btn');
      const on = !btn.textContent.includes('开');
      btn.textContent = '♨ 座椅加热: ' + (on ? '开' : '关');
      btn.style.background = on ? 'rgba(0,200,255,0.15)' : '';
    };
  });
})();