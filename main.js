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

const SYMBOL = "R_100";
const DERIV_APP_ID = 120975;
const DERIV_TOKEN = "S4B3gsvNAwpnHEQ";
const TRADING_TIMEFRAME_MIN = 10;
const HISTORY_COUNT = 4000;
const STATS_WINDOW = 100;

const MIN_CONFIDENCE = 50;
const MAX_DRAWDOWN = 100;
const MAX_OPEN_TRADES = 1;

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
let accumulatedLoss = 0.00;
let recoveryStep = 0;

let windowSignals = [];
let activeVirtualTrades = [];
let runningPositions = [];
let pendingOrders = {};

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
    activeVirtualTrades = [];

    for (let i = startIndex; i <= lastSafeIndex; i++) {
        const currentCandle = liveCandles[i];
        const highPrice = parseFloat(currentCandle.high);
        const lowPrice = parseFloat(currentCandle.low);

        for (let v = activeVirtualTrades.length - 1; v >= 0; v--) {
            const vt = activeVirtualTrades[v];
            
            if (vt.entryEpoch !== currentCandle.epoch) {
                vt.candlesElapsed++;
            }
            
            let maxProfit = 0;
            let minProfit = 0;
            let currentProfit = 0;

            if (vt.signal === "BUY") {
                maxProfit = ((highPrice - vt.entryPrice) / vt.entryPrice) * vt.stake * vt.multiplier - Math.abs(vt.spread);
                minProfit = ((lowPrice - vt.entryPrice) / vt.entryPrice) * vt.stake * vt.multiplier - Math.abs(vt.spread);
                currentProfit = ((parseFloat(currentCandle.close) - vt.entryPrice) / vt.entryPrice) * vt.stake * vt.multiplier - Math.abs(vt.spread);
            } else {
                maxProfit = ((vt.entryPrice - lowPrice) / vt.entryPrice) * vt.stake * vt.multiplier - Math.abs(vt.spread);
                minProfit = ((vt.entryPrice - highPrice) / vt.entryPrice) * vt.stake * vt.multiplier - Math.abs(vt.spread);
                currentProfit = ((vt.entryPrice - parseFloat(currentCandle.close)) / vt.entryPrice) * vt.stake * vt.multiplier - Math.abs(vt.spread);
            }

            let isWin = null;
            
            if (maxProfit >= vt.tp && minProfit <= -Math.abs(vt.sl)) {
                isWin = false;
            } else if (minProfit <= -Math.abs(vt.sl)) {
                isWin = false;
            } else if (maxProfit >= vt.tp) {
                isWin = true;
            } else if (vt.candlesElapsed >= vt.target) {
                if (currentProfit > 0) {
                    isWin = true;
                } else {
                    isWin = false;
                }
            }

            if (isWin !== null) {
                const wsItem = windowSignals.find(s => s.id === vt.id);
                if (wsItem) wsItem.result = isWin;
                activeVirtualTrades.splice(v, 1);
            }
        }

        const pastSubset = liveCandles.slice(0, i + 1);
        const result = await getAiPrediction(pastSubset);
        
        let finalSignal = "HOLD";

        if (result && (result.signal === "BUY" || result.signal === "SELL")) {
            const conf = parseFloat(result.confidence || 100);
            if (conf >= MIN_CONFIDENCE && activeVirtualTrades.length < MAX_OPEN_TRADES) {
                finalSignal = result.signal;
                const stake = parseFloat(result.stake);
                const multiplier = parseInt(result.multiplier);
                const tp = parseFloat(result.tp);
                const sl = parseFloat(result.sl);
                const target = parseInt(result.target || 1);
                const spread = parseFloat(result.spread || 0);
                const entryPrice = parseFloat(currentCandle.close);

                activeVirtualTrades.push({ 
                    id: currentCandle.epoch, entryEpoch: currentCandle.epoch, signal: finalSignal, entryPrice, stake, multiplier, tp, sl, target, spread, candlesElapsed: 0
                });
            }
        }

        windowSignals.push({ id: currentCandle.epoch, signal: finalSignal, result: null });
        if (windowSignals.length > STATS_WINDOW) windowSignals.shift();
        
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

function checkVirtualTrades(candle) {
    const highPrice = parseFloat(candle.high);
    const lowPrice = parseFloat(candle.low);
    const currentPrice = parseFloat(candle.close);

    for (let i = activeVirtualTrades.length - 1; i >= 0; i--) {
        const vt = activeVirtualTrades[i];
        
        let maxProfit = 0;
        let minProfit = 0;
        let currentProfit = 0;

        if (vt.signal === "BUY") {
            maxProfit = ((highPrice - vt.entryPrice) / vt.entryPrice) * vt.stake * vt.multiplier - Math.abs(vt.spread);
            minProfit = ((lowPrice - vt.entryPrice) / vt.entryPrice) * vt.stake * vt.multiplier - Math.abs(vt.spread);
            currentProfit = ((currentPrice - vt.entryPrice) / vt.entryPrice) * vt.stake * vt.multiplier - Math.abs(vt.spread);
        } else {
            maxProfit = ((vt.entryPrice - lowPrice) / vt.entryPrice) * vt.stake * vt.multiplier - Math.abs(vt.spread);
            minProfit = ((vt.entryPrice - highPrice) / vt.entryPrice) * vt.stake * vt.multiplier - Math.abs(vt.spread);
            currentProfit = ((vt.entryPrice - currentPrice) / vt.entryPrice) * vt.stake * vt.multiplier - Math.abs(vt.spread);
        }

        let isWin = null;
        
        if (maxProfit >= vt.tp && minProfit <= -Math.abs(vt.sl)) {
            isWin = false;
        } else if (minProfit <= -Math.abs(vt.sl)) {
            isWin = false;
        } else if (maxProfit >= vt.tp) {
            isWin = true;
        } else if (vt.candlesElapsed >= vt.target) {
             if (currentProfit > 0) {
                 isWin = true;
             } else {
                 isWin = false;
             }
        }

        if (isWin !== null) {
            const wsItem = windowSignals.find(s => s.id === vt.id);
            if (wsItem) wsItem.result = isWin;

            activeVirtualTrades.splice(i, 1);
            updateStatsUI();
        }
    }
}

async function runLiveAnalysis() {
    if (!isSystemReady || liveCandles.length < 2048) return;
    if (isRunningLive) return;
    isRunningLive = true;

    try {
        for (let i = 0; i < activeVirtualTrades.length; i++) {
            activeVirtualTrades[i].candlesElapsed++;
        }

        for (let i = 0; i < runningPositions.length; i++) {
            runningPositions[i].candlesElapsed++;
            
            if (runningPositions[i].candlesElapsed >= runningPositions[i].target && runningPositions[i].target > 0) {
                 if (ws && ws.readyState === WebSocket.OPEN) {
                     ws.send(JSON.stringify({ "sell": runningPositions[i].id, "price": 0 }));
                     runningPositions[i].target = 0;
                 }
            }
        }

        const result = await getAiPrediction(liveCandles);
        const lastCandle = liveCandles[liveCandles.length - 1];
        const currentPrice = parseFloat(lastCandle.close);
        
        let finalSignal = "HOLD";

        if (result && result.signal !== undefined) {
            const signal = result.signal;
            const confidence = parseFloat(result.confidence);
            const baseStake = parseFloat(result.stake);
            
            const actualStake = parseFloat((baseStake + (baseStake * 0.10 * recoveryStep)).toFixed(2));
            const scaleFactor = actualStake / baseStake;
            
            const multiplier = parseInt(result.multiplier);
            const tp = parseFloat((parseFloat(result.tp) * scaleFactor).toFixed(2));
            const sl = parseFloat((parseFloat(result.sl) * scaleFactor).toFixed(2));
            const target = parseInt(result.target || 1);
            const spread = parseFloat(result.spread || 0.10);

            if ((signal === "BUY" || signal === "SELL") && confidence > MIN_CONFIDENCE && runningPositions.length < MAX_OPEN_TRADES) {
                    sendToUI('bot-log', `[BOT] ${signal} (${confidence.toFixed(1)}%) | Price: ${currentPrice} | Stake: ${actualStake.toFixed(2)}`);
                    finalSignal = signal;

                    activeVirtualTrades.push({ 
                        id: lastCandle.epoch, entryEpoch: lastCandle.epoch, signal: finalSignal, entryPrice: currentPrice, stake: actualStake, multiplier, tp, sl, target, spread, candlesElapsed: 0
                    });

                    if (isTradingEnabled) {
                        placeDerivTrade(signal, currentPrice, actualStake, multiplier, tp, sl, target);
                    }
            } else {
                if (signal === "BUY" || signal === "SELL") {
                    sendToUI('bot-log', `[FILTER] ${signal} (${confidence.toFixed(1)}%) | Price: ${currentPrice}`);
                } else {
                    sendToUI('bot-log', `[FILTER] HOLD (${confidence.toFixed(1)}%) | Price: ${currentPrice}`);
                }
            }
        }

        windowSignals.push({ id: lastCandle.epoch, signal: finalSignal, result: null });
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
    accumulatedLoss = 0.00;
    recoveryStep = 0;
    isAnalyzingHistory = false;
    isRunningLive = false;
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
            if (msg.req_id && pendingOrders[msg.req_id]) {
                delete pendingOrders[msg.req_id];
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
                    lastProcessedEpoch = currentEpoch;
                    runLiveAnalysis();
                }
            }
            
            checkVirtualTrades(candle);
        }

        if (msg.msg_type === 'proposal_open_contract') {
            const contract = msg.proposal_open_contract;
            if (!contract || !contract.contract_id) return;
            const contractId = contract.contract_id;

            if (contract.is_sold) {
                const existingIndex = runningPositions.findIndex(p => p.id === contractId);
                if (existingIndex !== -1) {
                    runningPositions.splice(existingIndex, 1);

                    const profit = parseFloat(contract.profit);
                    let status = "BREAKEVEN";

                    if (profit > 0) {
                        status = "WIN";
                        realSessionTP++;
                        accumulatedLoss -= profit;
                        if (accumulatedLoss <= 0) {
                            accumulatedLoss = 0;
                            recoveryStep = 0;
                        }
                    } else if (profit < 0) {
                        status = "LOSS";
                        realSessionSL++;
                        accumulatedLoss += Math.abs(profit);
                        recoveryStep++;
                    }
                    
                    accumulatedLoss = parseFloat(accumulatedLoss.toFixed(2));
                    const profitText = (profit > 0 ? "+" : "") + profit.toFixed(2);
                    
                    updateStatsUI();
                    sendToUI('bot-log', `[CLOSED] ${status}: ${profitText} USD | ID: ${contractId}`);
                }
            } else {
                let existing = runningPositions.find(p => p.id === contractId);
                if (!existing) {
                    runningPositions.push({ id: contractId, type: contract.contract_type, target: 0, candlesElapsed: 0 });
                } else {
                    existing.type = contract.contract_type;
                }
            }
        }

        if (msg.msg_type === 'buy') {
            const contractId = msg.buy.contract_id;
            const reqId = msg.req_id;
            let targetValue = 0;

            if (reqId && pendingOrders[reqId]) {
                targetValue = pendingOrders[reqId].target;
                delete pendingOrders[reqId];
            }

            if (!runningPositions.find(p => p.id === contractId)) {
                runningPositions.push({ id: contractId, type: "UNKNOWN", target: targetValue, candlesElapsed: 0 });
            } else {
                const existing = runningPositions.find(p => p.id === contractId);
                existing.target = targetValue;
            }
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

function placeDerivTrade(signal, entryPrice, stake, multiplier, tp, sl, target) {
    if (!ws) return;

    const reqId = Date.now() + Math.floor(Math.random() * 1000);
    pendingOrders[reqId] = { target: target };

    const contractType = (signal === "BUY") ? "MULTUP" : "MULTDOWN";
    ws.send(JSON.stringify({
        "buy": 1,
        "price": 1000,
        "parameters": {
            "amount": stake,
            "basis": "stake",
            "contract_type": contractType,
            "currency": "USD",
            "symbol": SYMBOL,
            "multiplier": multiplier,
            "limit_order": {
                "stop_loss": sl,
                "take_profit": tp
            }
        },
        "req_id": reqId
    }));
    sendToUI('bot-log', `[ENTRY] ${contractType} | Price: ${entryPrice} | Limit: ${sl.toFixed(2)}/${tp.toFixed(2)}`);
}

function startDerivBot() { 
    isExplicitlyStopped = false;
    if (!isBotConnected) connectToDerivAPI(); 
}

ipcMain.on('toggle-trading-logic', (e, isActive) => {
    if (isActive && isCircuitTripped) {
        isCircuitTripped = false;
        initialBalance = sessionStats.balance;
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

setInterval(() => {
    if (isBotConnected && ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ "ping": 1 }));
    }
}, 10000);