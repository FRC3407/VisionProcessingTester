package org.usfirst.frc.team3407.vision;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
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

    private static final Scalar RED = new Scalar(0, 0, 255);
    private static final double[][] HSL = {
        {49.0, 57.0, 126.0},
        {20.0, 23.0, 90.0},
    };
    private static final Scalar[] COLORS = {
        new Scalar(0, 0, 255),
        new Scalar(255, 0, 0)
    };

    private static final double MIN_AREA = 1000;
    private static final double MAX_AREA = 4000;
    private static final double TARGET_RATIO = 5.0;
    private static final double TARGET_RATIO_OFFSET = 2.5;

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

        System.out.println(String.format("Starting video capture for %s: open=%s", fileName, videoCapture.isOpened()));
        Mat frame = new Mat();

        int frameCount = 0;
        while (videoCapture.read(frame)) {
            System.out.println("FrameCount=" + frameCount++);
            for (int i = 0;i < HSL.length;i++) {
                List<RotatedRect> rectangles = runPipeline(frame, HSL[i][0], HSL[i][1], HSL[i][2]);
                System.out.println(String.format("Found %s possible target matches", rectangles.size()));
                for (RotatedRect rectangle : rectangles) {
                    Point[] points = new Point[4];
                    rectangle.points(points);
                    for (int p = 0; p < 4; p++) {
                        Imgproc.line(frame, points[p], points[(p + 1) % 4], COLORS[i], 3);
                    }
                }
            }
            Imgcodecs.imwrite(outputFileName, frame);
            Thread.sleep(5000);
        }
    }

    private List<RotatedRect> runPipeline(Mat image, double minHue, double minSaturation, double minLuminance) {
        GripPipeline gripPipeline = executePipeline(image, minHue, minSaturation, minLuminance);
        return processPipelineOutputs(gripPipeline, MIN_AREA, MAX_AREA, TARGET_RATIO, TARGET_RATIO_OFFSET);
    }

    private GripPipeline executePipeline(Mat image, double minHue, double minSaturation, double minLuminance) {
        GripPipeline gripPipeline = new GripPipeline();
        gripPipeline.setMinHue(minHue);
        gripPipeline.setMinSaturation(minSaturation);
        gripPipeline.setMinLuminance(minLuminance);

        gripPipeline.process(image);

        return gripPipeline;
    }

    private List<RotatedRect> processPipelineOutputs(GripPipeline pipeline, double minArea, double maxArea,
                                                     double targetRatio, double targetRatioOffset) {
        List<MatOfPoint> contours = pipeline.findContoursOutput();
        System.out.println(String.format("Found %s contours", contours.size()));

        ArrayList<RotatedRect> filtered = new ArrayList<>();
        for(MatOfPoint contour : contours) {
            MatOfPoint2f matOfPoint2f = new MatOfPoint2f();
            contour.convertTo(matOfPoint2f, CvType.CV_32FC2);
            RotatedRect rr = Imgproc.minAreaRect(matOfPoint2f);
            double area = rr.size.area();
            double ratio = rr.size.height / rr.size.width;

            boolean inAreaRange = (area > minArea) && (area < maxArea);
            boolean inRatioRange = inRatioRange(ratio, targetRatio, targetRatioOffset);
            if (inAreaRange && inRatioRange) {
                System.out.println(String.format("Center=%s Area=%s Angle=%s Ratio=%s", rr.center, area, rr.angle,
                        ratio));
                filtered.add(rr);
            }
        }

        return filtered;
    }

    private boolean inRatioRange(double ratio, double target, double offset) {
        if (ratio >= 1) {
            double lower = target - offset;
            double upper = target + offset;
            return (ratio > lower) && (ratio < upper);
        } else {
            double lower = 1.0 / (target + offset);
            double upper = 1.0 / (target - offset);
            return (ratio > lower) && (ratio < upper);
        }
    }
}
