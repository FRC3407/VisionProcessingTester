package org.usfirst.frc.team3407.vision;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import java.io.File;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class Main {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    private static final Scalar BLUE = new Scalar(255, 0, 0);
    private static final Scalar GREEN = new Scalar(0, 255, 0);
    private static final Scalar RED = new Scalar(0, 0, 255);
    private static final Scalar BLACK = new Scalar(0, 0, 0);

    private static final Scalar[] COLORS = { RED, BLUE };

    private static final String OUTPUT_PATH_PREFIX = "C:\\Users\\jstho\\GRIP\\output\\out_";

    public static void main(String[] args) {
        new Main().run((args.length == 0)? null : args[0]);
    }

    public void run(String pathName) {
        try {
            Path path = FileSystems.getDefault().getPath(pathName);
            if (Files.isDirectory(path)) {
                File[] files = path.toFile().listFiles();
                for (File file : files) {
                    Path filePath = file.toPath();
                    if (Files.isRegularFile(filePath)) {
                        processFrames(filePath);
                    }
                }
            } else {
                processFrames(path);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void processFrames(Path path) throws Exception {
        String fileName = path.getFileName().toString();
        processFrames(path.toString(), OUTPUT_PATH_PREFIX  + fileName);
    }

    private void processFrames(String fileName, String outputFileName) throws Exception {
        VideoCapture videoCapture = (fileName == null) ? new VideoCapture(0) : new VideoCapture(fileName);
        HatchTargetRecognizer hatchTargetRecognizer = new HatchTargetRecognizer();

        System.out.println(String.format("\nStarting video capture for %s: open=%s", fileName, videoCapture.isOpened()));
        Mat frame = new Mat();

        int frameCount = 0;
        while (videoCapture.read(frame)) {
            System.out.println("\nFrameCount=" + frameCount++);
            System.out.println(String.format("Max-X is %s and Max-Y is %s", frame.width(), frame.height()));

            List<HatchTargetRecognizer.HatchTarget> hatchTargets = hatchTargetRecognizer.find(frame,
                    (r, i) -> drawRotatedRectangle(frame, r, COLORS[i]));

            int targetIndex = 0;
            for (HatchTargetRecognizer.HatchTarget hatchTarget : hatchTargets) {
                drawRotatedRectangle(frame, hatchTarget.getLeft(), BLACK);
                drawRotatedRectangle(frame, hatchTarget.getRight(), BLACK);
                double offset = hatchTarget.getOffset(frame.width());
                System.out.println(String.format("Target %s is %s pixels %s of center", targetIndex, Math.abs(offset),
                        (offset < 0) ? "left" : "right"));
                targetIndex++;
            }

            Imgcodecs.imwrite(outputFileName, frame);
            Thread.sleep(5000);
        }
    }

    private void drawRotatedRectangle(Mat image, RotatedRect rotatedRectangle, Scalar color) {
        Point[] points = new Point[4];
        rotatedRectangle.points(points);
        for (int p = 0; p < 4; p++) {
            Imgproc.line(image, points[p], points[(p + 1) % 4], color, 3);
        }

    }
}
