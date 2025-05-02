const express = require("express");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");
const morgan = require("morgan");
const app = express();
app.use(morgan("dev"));
const PORT = 3000;

const predict = [];

// สร้างโฟลเดอร์ Uploads ถ้ายังไม่มี
const uploadDir = "./Uploads";
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir);
}

// ตั้งค่า multer สำหรับจัดการไฟล์
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    cb(null, file.originalname);
  },
});

const upload = multer({
  storage: storage,
  limits: { fileSize: 5 * 1024 * 1024 }, // จำกัดขนาดไฟล์ 5MB
});

// Route สำหรับรับภาพและประมวลผลด้วย AI
app.post("/upload", upload.single("image"), (req, res) => {
  if (!req.file) {
    console.log("No image uploaded or invalid format");
    return res.status(400).send("No image uploaded");
  }

  const filename = req.file.filename;
  const filePath = path.join(uploadDir, filename);
  console.log(`Image saved as ${filename}, size: ${req.file.size} bytes`);

  // เรียกสคริปต์ Python เพื่อประมวลผลภาพ
  const pythonProcess = spawn("python", ["predict.py", filePath], {
    env: { ...process.env, PYTHONIOENCODING: "utf-8" },
  });

  let prediction = "";
  let errorOutput = "";

  pythonProcess.stdout.on("data", (data) => {
    prediction += data.toString();
  });

  pythonProcess.stderr.on("data", (data) => {
    const errorMessage = data.toString("utf8");
    if (!errorMessage.includes("Traceback")) {
      return;
    }
    errorOutput += errorMessage;
  });

  pythonProcess.on("close", (code) => {
    if (code === 0) {
      const pred = prediction.trim();
      console.log(`AI Prediction for ${filename}: ${pred}`);
      predict.push(pred);
      res.status(200).json({
        message: "Image saved",
        prediction: pred
      });
    } else {
      console.error(`Python script error: ${errorOutput}`);
      res.status(500).send(`Error processing image: ${errorOutput}`);
    }
  });
});

app.use("/uploads", express.static(path.join(__dirname, "Uploads")));

// Route สำหรับส่งข้อความและผลการทำนาย
app.get("/", (req, res) => {
  res.json({
    message: "Server is running. Use /upload to upload an image.",
    predictions: predict
  });
});

// เริ่มเซิร์ฟเวอร์
app.listen(PORT, () => {
  console.log(`Server running on http://10.0.0.112:${PORT}`);
});