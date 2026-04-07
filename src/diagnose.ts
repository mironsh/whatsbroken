import Anthropic from "@anthropic-ai/sdk";
import Groq from "groq-sdk";
import ffmpeg from "fluent-ffmpeg";
import fs from "fs";
import path from "path";
import readline from "readline";
import { fileURLToPath } from "url";
import dotenv from "dotenv";
import ffmpegPath from "ffmpeg-static";

dotenv.config({ path: ".env.local" });
ffmpeg.setFfmpegPath(ffmpegPath!);

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY! });
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY! });

// --- types ---
interface ApplianceIdentification {
    appliance: string;
    brand: string;
    observedSymptoms: string[];
}

interface RepairCost {
    diy: string;
    professional: string;
}

interface DiagnosisResult {
    appliance: string;
    likelyIssue: string;
    possibleCauses: string[];
    severity: "Low" | "Medium" | "High";
    recommendedAction: string;
    safetyWarning: string;
    estimatedRepairCost: RepairCost;
    clarifyingQuestions: string[];
}

// --- args ---
const args = process.argv.slice(2);
const videoFlagIndex = args.indexOf("--video");
if (videoFlagIndex === -1 || !args[videoFlagIndex + 1]) {
    console.error("Usage: npx tsx diagnose.ts --video <path-to-video>");
    process.exit(1);
}
const videoPath = path.resolve(args[videoFlagIndex + 1]);
if (!fs.existsSync(videoPath)) {
    console.error(`File not found: ${videoPath}`);
    process.exit(1);
}

// --- helpers ---
const tmpDir = path.join(__dirname, ".tmp");
if (!fs.existsSync(tmpDir)) fs.mkdirSync(tmpDir);

function cleanup(): void {
    fs.readdirSync(tmpDir).forEach((f) =>
        fs.unlinkSync(path.join(tmpDir, f))
    );
}

function extractFrames(videoPath: string, outputDir: string, prefix: string, fps = 0.33): Promise<string[]> {
    return new Promise((resolve, reject) => {
        const pattern = `${outputDir}/${prefix}-%04d.jpg`;
        ffmpeg(videoPath)
            .outputOptions([`-vf fps=${fps}`, "-q:v 3"])
            .output(pattern)
            .on("end", () => {
                const frames = fs
                    .readdirSync(outputDir)
                    .filter((f) => f.startsWith(prefix) && f.endsWith(".jpg"))
                    .sort()
                    .map((f) => path.join(outputDir, f));
                resolve(frames);
            })
            .on("error", (err) => reject(err))
            .run();
    });
}

function extractAudio(videoPath: string, outputPath: string): Promise<void> {
    return new Promise((resolve, reject) => {
        ffmpeg(videoPath)
            .outputOptions(["-ar 16000", "-ac 1", "-c:a pcm_s16le"])
            .output(outputPath)
            .on("end", () => resolve())
            .on("error", reject)
            .run();
    });
}

async function transcribeAudio(audioPath: string): Promise<string> {
    const audioBuffer = fs.readFileSync(audioPath);
    const file = new File([audioBuffer], "audio.wav", { type: "audio/wav" });
    const result = await groq.audio.transcriptions.create({
        file,
        model: "whisper-large-v3-turbo",
    });
    return result.text?.trim() ?? "";
}

function frameToBase64(framePath: string): string {
    return fs.readFileSync(framePath).toString("base64");
}

function buildImageContent(frames: string[], cache = false) {
    const selectedFrames = frames.length > 10
        ? frames.filter((_, i) => i % Math.ceil(frames.length / 10) === 0).slice(0, 10)
        : frames;
    return selectedFrames.map((f, i) => ({
        type: "image" as const,
        source: {
            type: "base64" as const,
            media_type: "image/jpeg" as const,
            data: frameToBase64(f),
        },
        ...(cache && i === selectedFrames.length - 1 ? { cache_control: { type: "ephemeral" as const } } : {}),
    }));
}

async function processVideo(videoPath: string, prefix: string): Promise<{ frames: string[]; transcript: string }> {
    const frames = await extractFrames(videoPath, tmpDir, prefix);
    const audioPath = path.join(tmpDir, `${prefix}-audio.wav`);
    await extractAudio(videoPath, audioPath);
    const transcript = await transcribeAudio(audioPath);
    return { frames, transcript };
}

function ask(prompt: string): Promise<string> {
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    return new Promise((resolve) => {
        rl.question(prompt, (answer) => {
            rl.close();
            resolve(answer.trim());
        });
    });
}

async function identifyAppliance(frames: string[], transcript: string): Promise<{ result: ApplianceIdentification; usage: Anthropic.Usage }> {
    const imageContent = buildImageContent(frames, true);
    const textContent = {
        type: "text" as const,
        text: transcript
            ? `The user says: "${transcript}"\n\nIdentify the appliance and list every visible or audible symptom.`
            : "Identify the appliance and list every visible symptom.",
    };

    const response = await anthropic.messages.create({
        model: "claude-sonnet-4-6",
        max_tokens: 512,
        system: `You are an expert appliance technician. Your only job right now is to identify what appliance is shown and list the symptoms you observe — do NOT diagnose yet.

Respond only with a valid JSON object, no markdown, no preamble:
{
  "appliance": "appliance type (e.g. washing machine, refrigerator, dishwasher)",
  "brand": "brand name if visible, otherwise Unknown",
  "observedSymptoms": ["symptom 1", "symptom 2"]
}`,
        messages: [{ role: "user", content: [...imageContent, textContent] }],
    });

    const raw = response.content[0].type === "text" ? response.content[0].text : "";
    return { result: JSON.parse(raw) as ApplianceIdentification, usage: response.usage };
}

function buildDiagnoseSystemPrompt(id: ApplianceIdentification): string {
    return `You are a specialist repair technician for ${id.appliance}s. Your task is to diagnose the root cause and advise the user.

When forming clarifyingQuestions, you may ask the user to perform safe physical actions and record a new video — for example: "Flip the unit upside-down and show me the bottom label", "Turn it on and hold the phone near it so I can hear the sound it makes", or "Open the door/panel and show me inside". Only suggest actions that are safe (no exposed live wiring, no gas lines, no heavy components).

Respond only with a valid JSON object in this exact shape, no markdown, no preamble:
{
  "appliance": "full description including brand",
  "likelyIssue": "1-2 sentence summary of the most probable fault",
  "possibleCauses": ["most likely cause", "second possibility", "third possibility"],
  "severity": "Low" | "Medium" | "High",
  "recommendedAction": "step-by-step DIY fix if safe, otherwise advise calling a professional",
  "safetyWarning": "specific electrical/gas/water hazards for this appliance, or None",
  "estimatedRepairCost": { "diy": "e.g. $0-20 (part name)", "professional": "e.g. $80-150" },
  "clarifyingQuestions": ["question or safe action that would increase diagnostic confidence"] or [] if diagnosis is already certain
}`;
}

async function diagnose(
    messages: Anthropic.MessageParam[],
    systemPrompt: string
): Promise<{ result: DiagnosisResult; usage: Anthropic.Usage }> {
    const response = await anthropic.messages.create({
        model: "claude-sonnet-4-6",
        max_tokens: 10000,
        thinking: { type: "enabled", budget_tokens: 8000 },
        system: systemPrompt,
        messages,
    });

    const raw = response.content.find((b) => b.type === "text")?.text ?? "";
    return { result: JSON.parse(raw) as DiagnosisResult, usage: response.usage };
}

function buildInitialMessage(frames: string[], transcript: string, id: ApplianceIdentification): Anthropic.MessageParam {
    const imageContent = buildImageContent(frames, true);
    const symptomList = id.observedSymptoms.map((s) => `• ${s}`).join("\n");
    const text = [
        `Appliance: ${id.appliance} (${id.brand})`,
        `Observed symptoms:\n${symptomList}`,
        transcript ? `User description: "${transcript}"` : "",
        "\nProvide the full structured diagnosis.",
    ].filter(Boolean).join("\n");

    return { role: "user", content: [...imageContent, { type: "text", text }] };
}

function printDiagnosis(d: DiagnosisResult): void {
    const severityColor = { Low: "\x1b[32m", Medium: "\x1b[33m", High: "\x1b[31m" }[d.severity];
    const reset = "\x1b[0m";

    console.log("\n✅ DIAGNOSIS");
    console.log("============");
    console.log(`🔧 Appliance:   ${d.appliance}`);
    console.log(`🩺 Issue:       ${d.likelyIssue}`);
    console.log(`📋 Causes:`);
    d.possibleCauses.forEach((c) => console.log(`   • ${c}`));
    console.log(`⚠️  Severity:    ${severityColor}${d.severity}${reset}`);
    console.log(`🛠️  Action:      ${d.recommendedAction}`);
    console.log(`💰 Cost:        DIY ${d.estimatedRepairCost.diy} · Professional ${d.estimatedRepairCost.professional}`);
    console.log(`🚨 Safety:      ${d.safetyWarning}\n`);
}

// --- main ---
async function main(): Promise<void> {
    console.log("\n🔧 WhatsBroken — Appliance Diagnosis");
    console.log("=====================================");
    console.log(`📹 Video: ${videoPath}\n`);

    try {
        console.log("⏳ Extracting frames & audio...");
        const { frames, transcript } = await processVideo(videoPath, "frame");
        console.log(`   ${frames.length} frames extracted`);
        if (transcript) console.log(`   Transcript: "${transcript}"`);
        else console.log("   No speech detected");

        console.log("⏳ Identifying appliance...");
        const { result: id, usage: usage1 } = await identifyAppliance(frames, transcript);
        console.log(`   ${id.appliance} (${id.brand}) — ${id.observedSymptoms.length} symptom(s) detected`);
        console.log(`   Tokens: ${usage1.input_tokens} in / ${usage1.output_tokens} out (${usage1.cache_creation_input_tokens ?? 0} cached)`);

        const systemPrompt = buildDiagnoseSystemPrompt(id);
        const conversationMessages: Anthropic.MessageParam[] = [buildInitialMessage(frames, transcript, id)];

        console.log("⏳ Diagnosing (extended thinking)...");
        let { result: diagnosis, usage: usage2 } = await diagnose(conversationMessages, systemPrompt);
        console.log(`   Tokens: ${usage2.input_tokens} in / ${usage2.output_tokens} out (${usage2.cache_read_input_tokens ?? 0} from cache)`);
        printDiagnosis(diagnosis);

        // --- follow-up loop ---
        let round = 1;
        while (diagnosis.clarifyingQuestions.length > 0) {
            console.log("❓ To improve this diagnosis:");
            diagnosis.clarifyingQuestions.forEach((q) => console.log(`   • ${q}`));
            console.log();

            const textAnswer = await ask("Your answer (or press Enter to skip): ");
            const videoInput = await ask("Follow-up video path (or press Enter to skip): ");

            if (!textAnswer && !videoInput) break;

            const followUpContent: Anthropic.MessageParam["content"] = [];

            if (videoInput) {
                if (!fs.existsSync(videoInput)) {
                    console.error(`   File not found: ${videoInput}`);
                } else {
                    console.log("⏳ Processing follow-up video...");
                    const { frames: newFrames, transcript: newTranscript } = await processVideo(videoInput, `frame-r${round}`);
                    console.log(`   ${newFrames.length} frames extracted`);
                    if (newTranscript) console.log(`   Transcript: "${newTranscript}"`);
                    followUpContent.push(...buildImageContent(newFrames));
                    if (newTranscript) followUpContent.push({ type: "text", text: `Follow-up video transcript: "${newTranscript}"` });
                }
            }

            if (textAnswer) followUpContent.push({ type: "text", text: textAnswer });

            if (followUpContent.length === 0) break;

            // Append assistant response + new user message to conversation
            conversationMessages.push({ role: "assistant", content: JSON.stringify(diagnosis) });
            conversationMessages.push({ role: "user", content: followUpContent });

            console.log("⏳ Re-diagnosing...");
            ({ result: diagnosis, usage: usage2 } = await diagnose(conversationMessages, systemPrompt));
            console.log(`   Tokens: ${usage2.input_tokens} in / ${usage2.output_tokens} out`);
            printDiagnosis(diagnosis);
            round++;
        }
    } catch (err) {
        console.error("\n❌ Error:", err instanceof Error ? err.message : err);
    } finally {
        cleanup();
    }
}

main();
