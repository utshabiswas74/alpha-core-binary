const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const WebSocket = require('ws');

const isDev = !app.isPackaged;
const RESOURCES_PATH = isDev ? __dirname : process.resourcesPath;
const ROOT_DIR = __dirname;

const PUBLIC_DIR = path.join(ROOT_DIR, 'public');
const ICON_PATH = path.join(PUBLIC_DIR, 'icon.ico');
const INDEX_PATH = path.join(PUBLIC_DIR, 'index.html');
const ENGINE_PATH = path.join(RESOURCES_PATH, 'bin', 'engine.exe');

const DERIV_TOKEN = "S4B3gsvNAwpnHEQ";
const TRADING_TIMEFRAME_MIN = 1;
const DERIV_APP_ID = 120975;
const HISTORY_COUNT = 4000;
const STATS_WINDOW = 100;
const BASE_STAKE = 0.40;
const MAX_DRAWDOWN = 80;
const SYMBOL = "R_100";

let mainWindow;
let ws = null;
let liveCandles = [];
let isBotConnected = false;
let isTradingEnabled = false;
let isExplicitlyStopped = false;
let isSystemReady = false;
let isCircuitTripped = false;
let isAutoReconnecting = false;
let isAnalyzingHistory = false;
let isRunningLive = false;

let lastProcessedEpoch = 0;
let initialBalance = null;
let targetBalance = null;
let accumulatedLoss = 0.00;
let recoveryStep = 0;
let systemLockEpoch = 0;

let windowSignals = [];
let runningPositions = [];

let realSessionTP = 0;
let realSessionSL = 0;

let sessionStats = {
    balance: 0.00,
    totalProfit: 0.00,
    accuracy: "0/0",
    winRate: "0/0"
};

function sendToUI(channel, data) {
    if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send(channel, data);
    }
}

function updateStatsUI() {
    const validTrades = windowSignals.filter(s => (s.signal === 'BUY' || s.signal === 'SELL') && s.result !== null);
    const windowTP = validTrades.filter(s => s.result === true).length;
    const totalWindow = validTrades.length;
    
    sessionStats.accuracy = `${windowTP}/${totalWindow}`;

    const totalReal = realSessionTP + realSessionSL;
    sessionStats.winRate = `${realSessionTP}/${totalReal}`;

    sendToUI('bot-data-update', {
        balance: sessionStats.balance.toFixed(2),
        pnl: sessionStats.totalProfit.toFixed(2),
        accuracy: sessionStats.accuracy,
        winRate: sessionStats.winRate
    });
}

function formatTime(epoch) {
    if (!epoch) return { date: "0000.00.00", time: "00:00" };
    const d = new Date(epoch * 1000);
    const dateStr = d.toISOString().split('T')[0].replace(/-/g, '.');
    const timeStr = d.toTimeString().split(' ')[0].substring(0, 5);
    return { date: dateStr, time: timeStr };
}

function getAiPrediction(candleSubset) {
    return new Promise((resolve) => {
        if (candleSubset.length < 2048) { resolve(null); return; }

        let csvData = "";
        
        for (let i = 0; i < candleSubset.length - 1; i++) {
            const c = candleSubset[i];
            const timeVal = c.epoch || c.open_time;
            const t = formatTime(timeVal);
            csvData += `${t.date} ${t.time} ${c.open} ${c.high} ${c.low} ${c.close} 100\n`;
        }

        const aiProcess = spawn(ENGINE_PATH, [], { cwd: RESOURCES_PATH });

        aiProcess.stdin.on('error', () => { resolve(null); });

        let jsonOutput = "";
        aiProcess.stdout.on('data', (data) => { jsonOutput += data.toString(); });

        try {
            aiProcess.stdin.write(csvData);
            aiProcess.stdin.end();
        } catch (e) {
            aiProcess.kill();
            resolve(null);
        }

        aiProcess.on('close', (code) => {
            if (code === 0 && jsonOutput.includes("JSON_START")) {
                try {
                    const cleanJson = jsonOutput.split("JSON_START")[1].split("JSON_END")[0].trim();
                    resolve(JSON.parse(cleanJson));
                } catch (e) { resolve(null); }
            } else { resolve(null); }
        });
    });
}

async function analyzeHistoryBatch() {
    if (isAnalyzingHistory) return;
    if (liveCandles.length < 2048) return;
    
    isAnalyzingHistory = true;

    if (!isAutoReconnecting) {
        sendToUI('bot-log', `[INIT] Running Backtest, ${STATS_WINDOW} Signals...`);
    }

    const lastSafeIndex = liveCandles.length - 2;
    const startIndex = Math.max(0, lastSafeIndex - STATS_WINDOW);

    windowSignals = [];
    systemLockEpoch = 0;

    for (let i = startIndex; i <= lastSafeIndex; i++) {
        const currentCandle = liveCandles[i];
        const pastSubset = liveCandles.slice(0, i + 1);
        const result = await getAiPrediction(pastSubset);
        
        let finalSignal = "HOLD";
        let isWin = null;
        let targetEpoch = 0;
        let entryPrice = parseFloat(currentCandle.close);

        if (result && (result.signal === "BUY" || result.signal === "SELL")) {
            const target = result.target ? parseInt(result.target) : 1;
            const tempTargetEpoch = currentCandle.epoch + (target * TRADING_TIMEFRAME_MIN * 60);
            
            if (currentCandle.epoch >= systemLockEpoch) {
                finalSignal = result.signal;
                targetEpoch = tempTargetEpoch;
                systemLockEpoch = targetEpoch + (TRADING_TIMEFRAME_MIN * 60);
                
                if (i + target < liveCandles.length) {
                    const exitPrice = parseFloat(liveCandles[i + target].close);
                    if (finalSignal === "BUY") {
                        isWin = exitPrice > entryPrice;
                    } else if (finalSignal === "SELL") {
                        isWin = exitPrice < entryPrice;
                    }
                }
            }
        }

        if (targetEpoch === 0) {
            targetEpoch = currentCandle.epoch + (TRADING_TIMEFRAME_MIN * 60);
        }

        windowSignals.push({ 
            id: currentCandle.epoch, 
            signal: finalSignal, 
            result: isWin,
            entryPrice: entryPrice,
            targetEpoch: targetEpoch,
            isRealTrade: false
        });
        if (windowSignals.length > STATS_WINDOW) windowSignals.shift();
        
        updateStatsUI();

        if (i % 10 === 0) await new Promise(r => setTimeout(r, 10));
    }

    isSystemReady = true;
    
    if (!isAutoReconnecting) {
        sendToUI('bot-log', `[INIT] Sync Complete, Success.`);
    } else {
        isAutoReconnecting = false;
    }
    
    updateStatsUI();
    isAnalyzingHistory = false;
}

async function runLiveAnalysis() {
    if (!isSystemReady || liveCandles.length < 2048) return;
    if (isRunningLive) return;
    isRunningLive = true;

    try {
        const result = await getAiPrediction(liveCandles);
        const lastCandle = liveCandles[liveCandles.length - 1];
        const currentPrice = parseFloat(lastCandle.close);
        
        let finalSignal = "HOLD";
        let isRealTrade = false;
        let targetEpoch = 0;

        if (result && result.signal !== undefined) {
            const signal = result.signal;
            const confidence = parseFloat(result.confidence || 0);
            const target = result.target ? parseInt(result.target) : 1;
            const tempTargetEpoch = lastCandle.epoch + (target * TRADING_TIMEFRAME_MIN * 60);
            
            const actualStake = parseFloat((BASE_STAKE + (BASE_STAKE * 0.25 * recoveryStep)).toFixed(2));

            if ((signal === "BUY" || signal === "SELL") && lastCandle.epoch >= systemLockEpoch) {
                sendToUI('bot-log', `[BOT] ${signal} (${confidence.toFixed(1)}%) | Price: ${currentPrice.toFixed(2)} | Stake: ${actualStake.toFixed(2)}`);
                finalSignal = signal;
                targetEpoch = tempTargetEpoch;
                systemLockEpoch = targetEpoch + (TRADING_TIMEFRAME_MIN * 60);

                if (isTradingEnabled) {
                    isRealTrade = true;
                    placeDerivTrade(signal, currentPrice, actualStake, lastCandle.epoch, target);
                }
            } else {
                if (signal === "BUY" || signal === "SELL") {
                    sendToUI('bot-log', `[FILTER] ${signal} (${confidence.toFixed(1)}%) | Price: ${currentPrice.toFixed(2)} | Stake: ${actualStake.toFixed(2)}`);
                } else {
                    sendToUI('bot-log', `[FILTER] HOLD (${confidence.toFixed(1)}%) | Price: ${currentPrice.toFixed(2)} | Stake: ${actualStake.toFixed(2)}`);
                }
            }
        } else {
            targetEpoch = lastCandle.epoch + (TRADING_TIMEFRAME_MIN * 60);
        }

        windowSignals.push({ 
            id: lastCandle.epoch, 
            signal: finalSignal, 
            result: null,
            entryPrice: currentPrice,
            targetEpoch: targetEpoch,
            isRealTrade: isRealTrade
        });
        if (windowSignals.length > STATS_WINDOW) windowSignals.shift();

        updateStatsUI();
    } finally {
        isRunningLive = false;
    }
}

function stopDerivBot() {
    isBotConnected = false;
    isTradingEnabled = false;
    isSystemReady = false;
    initialBalance = null;
    targetBalance = null;
    accumulatedLoss = 0.00;
    recoveryStep = 0;
    systemLockEpoch = 0;
    isAnalyzingHistory = false;
    isRunningLive = false;
    runningPositions = [];
    if (ws) {
        ws.terminate();
        ws = null;
    }
    sendToUI('bot-status', 'stopped');
    sendToUI('stop-triggered', true);
}

function connectToDerivAPI() {
    if (ws && ws.readyState === WebSocket.OPEN) return;

    if (!isAutoReconnecting) {
        sendToUI('sys_log', { msg: 'Connecting to Deriv Server (USA)' });
    }

    try {
        ws = new WebSocket(`wss://ws.binaryws.com/websockets/v3?app_id=${DERIV_APP_ID}`);
    } catch (e) { 
        if (!isExplicitlyStopped) setTimeout(connectToDerivAPI, 10000);
        return; 
    }

    ws.on('open', () => {
        isBotConnected = true;
        sendToUI('bot-status', 'running');
        ws.send(JSON.stringify({ "authorize": DERIV_TOKEN }));
    });

    ws.on('message', (data) => {
        let msg;
        try {
            msg = JSON.parse(data);
        } catch (e) {
            return;
        }

        if (msg.error) {
            if (msg.req_id) {
                const pendingIndex = runningPositions.findIndex(p => p.reqId === msg.req_id);
                if (pendingIndex !== -1) {
                    runningPositions.splice(pendingIndex, 1);
                }
            }

            if (msg.error.code === 'NoOpenPosition' || msg.error.code === 'BetExpired' || msg.error.code === 'InvalidState' || msg.error.code === 'InvalidContractUpdate') {
                return;
            }

            sendToUI('bot-log', `[DERIV ERROR] ${msg.error.code}: ${msg.error.message}`);
            if (msg.error.code === 'InvalidToken' || msg.error.code === 'AuthorizationRequired') {
                isExplicitlyStopped = true;
                stopDerivBot();
            }
            return;
        }

        if (msg.msg_type === 'authorize') {
            if (!isAutoReconnecting) {
                sendToUI('sys_log', { msg: `Auth Success: ${msg.authorize.email}` });
            }

            sessionStats.balance = parseFloat(msg.authorize.balance);
            if (initialBalance === null) {
                initialBalance = sessionStats.balance;
                targetBalance = sessionStats.balance;
            }
            sessionStats.totalProfit = sessionStats.balance - initialBalance;
            updateStatsUI();

            ws.send(JSON.stringify({ "balance": 1, "subscribe": 1 }));
            ws.send(JSON.stringify({ "proposal_open_contract": 1, "subscribe": 1 }));

            ws.send(JSON.stringify({
                "ticks_history": SYMBOL,
                "adjust_start_time": 1,
                "count": HISTORY_COUNT,
                "end": "latest",
                "start": 1,
                "style": "candles",
                "granularity": TRADING_TIMEFRAME_MIN * 60,
                "subscribe": 1
            }));
        }

        if (msg.msg_type === 'balance') {
            sessionStats.balance = parseFloat(msg.balance.balance);
            
            if (initialBalance !== null) {
                sessionStats.totalProfit = sessionStats.balance - initialBalance;

                if (sessionStats.balance >= targetBalance) {
                    targetBalance = sessionStats.balance;
                    accumulatedLoss = 0;
                    recoveryStep = 0;
                } else {
                    let currentLoss = parseFloat((targetBalance - sessionStats.balance).toFixed(2));
                    if (currentLoss > accumulatedLoss) {
                        recoveryStep++;
                        accumulatedLoss = currentLoss;
                    }
                }
            }

            if (initialBalance !== null && (initialBalance - sessionStats.balance) >= MAX_DRAWDOWN && !isCircuitTripped && isTradingEnabled) {
                isCircuitTripped = true;
                isTradingEnabled = false;
                sendToUI('bot-log', `TRADING PAUSED. AI logic cancelling...`);
                sendToUI('stop-triggered', true);
            }

            updateStatsUI();
        }

        if (msg.msg_type === 'candles') {
            liveCandles = msg.candles;
            if (!isAutoReconnecting) {
                sendToUI('sys_log', { msg: `Monitoring ${SYMBOL} (M${TRADING_TIMEFRAME_MIN}), History Syncing...` });
            }
            analyzeHistoryBatch();
        }

        if (msg.msg_type === 'ohlc') {
            const candle = msg.ohlc;
            candle.epoch = candle.open_time;

            const currentPrice = parseFloat(candle.close);
            const currentEpoch = candle.open_time;

            sendToUI('price-update', currentPrice);

            const lastStored = liveCandles[liveCandles.length - 1];

            if (lastStored && lastStored.epoch === candle.epoch) {
                liveCandles[liveCandles.length - 1] = candle;
            } else {
                liveCandles.push(candle);
                if (liveCandles.length > HISTORY_COUNT) liveCandles.shift();

                if (currentEpoch > lastProcessedEpoch) {
                    if (lastProcessedEpoch !== 0) {
                        const closedCandle = liveCandles[liveCandles.length - 2];
                        if (closedCandle) {
                            const closedEpoch = closedCandle.epoch;
                            const closedPrice = parseFloat(closedCandle.close);
                            const actualCloseTime = closedEpoch + (TRADING_TIMEFRAME_MIN * 60);

                            windowSignals.forEach(s => {
                                if (s.result === null && s.targetEpoch <= actualCloseTime && (s.signal === "BUY" || s.signal === "SELL")) {
                                    if (s.signal === "BUY") {
                                        s.result = closedPrice > s.entryPrice;
                                    } else if (s.signal === "SELL") {
                                        s.result = closedPrice < s.entryPrice;
                                    }

                                    if (s.isRealTrade && s.result !== null) {
                                        if (s.result === true) {
                                            realSessionTP++;
                                        } else {
                                            realSessionSL++;
                                        }
                                    }
                                }
                            });
                            updateStatsUI();
                        }
                    }

                    lastProcessedEpoch = currentEpoch;
                    runLiveAnalysis();
                }
            }
        }

        if (msg.msg_type === 'proposal_open_contract') {
            const contract = msg.proposal_open_contract;
            if (!contract || !contract.contract_id) return;
            const contractId = contract.contract_id;

            if (contract.is_sold) {
                const pos = runningPositions.find(p => p.id === contractId);
                if (pos) {
                    runningPositions = runningPositions.filter(p => p.id !== contractId);

                    const profit = parseFloat(contract.profit);
                    let status = "BREAKEVEN";

                    if (profit > 0) {
                        status = "WIN";
                    } else if (profit < 0) {
                        status = "LOSS";
                    }

                    const profitText = (profit > 0 ? "+" : "") + profit.toFixed(2);
                    
                    updateStatsUI();
                    sendToUI('bot-log', `[CLOSED] ${status}: ${profitText} USD | ID: ${contractId}`);
                }
            } else {
                let existing = runningPositions.find(p => p.id === contractId);
                if (existing) {
                    existing.type = contract.contract_type;
                }
            }
        }

        if (msg.msg_type === 'buy') {
            const contractId = msg.buy.contract_id;
            const reqId = msg.req_id;

            let existing = runningPositions.find(p => p.id === contractId);
            
            if (!existing && reqId) {
                existing = runningPositions.find(p => p.reqId === reqId);
            }
            
            if (!existing) {
                existing = runningPositions.find(p => p.id === null);
            }

            if (existing) {
                existing.id = contractId;
                if (reqId) existing.reqId = reqId;
            } else {
                runningPositions.push({ reqId: reqId || null, id: contractId, signalId: null, type: "UNKNOWN" });
            }

            let uniquePositions = [];
            runningPositions.forEach(pos => {
                if (pos.id === null) {
                    uniquePositions.push(pos);
                } else {
                    let existingUnique = uniquePositions.find(u => u.id === pos.id);
                    if (!existingUnique) {
                        uniquePositions.push(pos);
                    } else {
                        if (pos.reqId) existingUnique.reqId = pos.reqId;
                        if (pos.signalId) existingUnique.signalId = pos.signalId;
                        if (pos.type !== "UNKNOWN") existingUnique.type = pos.type;
                    }
                }
            });
            runningPositions = uniquePositions;
            
            sendToUI('bot-log', `[DERIV] Order Filled | ID: ${contractId}`);
        }
    });

    ws.on('close', () => {
        isBotConnected = false;
        isSystemReady = false;
        if (!isExplicitlyStopped) {
            isAutoReconnecting = true;
            sendToUI('bot-status', 'stopped');
            ws = null;
            setTimeout(connectToDerivAPI, 10000);
        } else {
            stopDerivBot();
        }
    });

    ws.on('error', () => {
        if (ws) ws.terminate();
    });
}

function placeDerivTrade(signal, entryPrice, stake, signalId, targetCandles) {
    if (!ws) return;

    const reqId = Date.now() + Math.floor(Math.random() * 1000);
    const contractType = (signal === "BUY") ? "CALL" : "PUT";
    const durationMinutes = TRADING_TIMEFRAME_MIN * targetCandles;

    runningPositions.push({ reqId: reqId, id: null, signalId: signalId, type: "UNKNOWN" });

    ws.send(JSON.stringify({
        "buy": 1,
        "price": 1000,
        "parameters": {
            "amount": stake,
            "basis": "stake",
            "contract_type": contractType,
            "currency": "USD",
            "symbol": SYMBOL,
            "duration": durationMinutes,
            "duration_unit": "m"
        },
        "req_id": reqId
    }));
    sendToUI('bot-log', `[ENTRY] ${contractType} | Price: ${entryPrice.toFixed(2)} | Stake: ${stake.toFixed(2)}`);
}

function startDerivBot() { 
    isExplicitlyStopped = false;
    if (!isBotConnected) connectToDerivAPI(); 
}

ipcMain.on('toggle-trading-logic', (e, isActive) => {
    if (isActive && isCircuitTripped) {
        isCircuitTripped = false;
        initialBalance = sessionStats.balance;
        targetBalance = sessionStats.balance;
    }
    isTradingEnabled = isActive;
});

ipcMain.on('bot-command', (e, cmd) => { 
    if (cmd === 'start') {
        startDerivBot();
    } else {
        isExplicitlyStopped = true;
        stopDerivBot();
    }
});

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1280, height: 850, title: "ALPHA CORE // CNN M" + TRADING_TIMEFRAME_MIN, icon: ICON_PATH,
        backgroundColor: '#050608', autoHideMenuBar: true,
        webPreferences: { nodeIntegration: true, contextIsolation: false }
    });
    mainWindow.setMenu(null);
    if (fs.existsSync(INDEX_PATH)) mainWindow.loadFile(INDEX_PATH);
    mainWindow.once('ready-to-show', () => { mainWindow.show(); });
    mainWindow.on('closed', () => { 
        isExplicitlyStopped = true;
        stopDerivBot(); 
        app.quit(); 
    });
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => { 
    if (process.platform !== 'darwin') app.quit(); 
});