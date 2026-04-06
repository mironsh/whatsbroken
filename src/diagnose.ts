import Anthropic from "@anthropic-ai/sdk";
import Groq from "groq-sdk";
import ffmpeg from "fluent-ffmpeg";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";
import ffmpegPath from "ffmpeg-static";

dotenv.config({ path: ".env.local" });
ffmpeg.setFfmpegPath(ffmpegPath!);

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY! });
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY! });

// --- types ---
interface DiagnosisResult {
    appliance: string;
    likelyIssue: string;
    possibleCauses: string[];
    severity: "Low" | "Medium" | "High";
    recommendedAction: string;
    safetyWarning: string;
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

function extractFrames(videoPath: string, outputDir: string, fps = 0.33): Promise<string[]> {
    return new Promise((resolve, reject) => {
        const pattern = outputDir + "/frame-%04d.jpg";
        ffmpeg(videoPath)
            .outputOptions([`-vf fps=${fps}`, "-q:v 3"])
            .output(pattern)
            .on("end", () => {
                const frames = fs
                    .readdirSync(outputDir)
                    .filter((f) => f.endsWith(".jpg"))
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

async function diagnose(frames: string[], transcript: string): Promise<DiagnosisResult> {
    const selectedFrames = frames.length > 10
        ? frames.filter((_, i) => i % Math.ceil(frames.length / 10) === 0).slice(0, 10)
        : frames;

    const imageContent = selectedFrames.map((f) => ({
        type: "image" as const,
        source: {
            type: "base64" as const,
            media_type: "image/jpeg" as const,
            data: frameToBase64(f),
        },
    }));

    const textContent = {
        type: "text" as const,
        text: transcript
            ? `The user described the issue as follows: "${transcript}"\n\nBased on the video frames and description above, provide a structured appliance diagnosis.`
            : "Based on the video frames above, provide a structured appliance diagnosis.",
    };

    const systemPrompt = `You are an expert appliance repair technician. Analyze the provided video frames and any user description to diagnose the appliance issue.

Respond only with a valid JSON object in this exact shape, no markdown, no preamble:
{
  "appliance": "type and brand if visible",
  "likelyIssue": "1-2 sentence summary",
  "possibleCauses": ["cause 1", "cause 2", "cause 3"],
  "severity": "Low" | "Medium" | "High",
  "recommendedAction": "DIY steps if safe, or advise calling a professional",
  "safetyWarning": "any electrical/gas/water hazards, or None"
}`;

    const response = await anthropic.messages.create({
        model: "claude-sonnet-4-6",
        max_tokens: 1024,
        system: systemPrompt,
        messages: [
            {
                role: "user",
                content: [...imageContent, textContent],
            },
        ],
    });

    const raw = response.content[0].type === "text" ? response.content[0].text : "";
    return JSON.parse(raw) as DiagnosisResult;
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
    console.log(`🚨 Safety:      ${d.safetyWarning}\n`);
}

// --- main ---
async function main(): Promise<void> {
    console.log("\n🔧 WhatsBroken — Appliance Diagnosis");
    console.log("=====================================");
    console.log(`📹 Video: ${videoPath}\n`);

    try {
        console.log("⏳ Extracting frames...");
        const frames = await extractFrames(videoPath, tmpDir);
        console.log(`   ${frames.length} frames extracted`);

        console.log("⏳ Extracting audio...");
        const audioPath = path.join(tmpDir, "audio.wav");
        await extractAudio(videoPath, audioPath);

        console.log("⏳ Transcribing audio...");
        const transcript = await transcribeAudio(audioPath);
        if (transcript) {
            console.log(`   Transcript: "${transcript}"`);
        } else {
            console.log("   No speech detected");
        }

        console.log("⏳ Analyzing with Claude...");
        const diagnosis = await diagnose(frames, transcript);
        printDiagnosis(diagnosis);
    } catch (err) {
        console.error("\n❌ Error:", err instanceof Error ? err.message : err);
    } finally {
        cleanup();
    }
}

main();