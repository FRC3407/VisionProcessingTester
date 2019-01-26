package org.usfirst.frc.team3407.vision;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import java.nio.file.FileSystems;
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

    public static void main(String[] args) {
        new Main().run((args.length == 0)? null : args[0]);
    }

    public void run(String fileName) {
        try {
            Path file = FileSystems.getDefault().getPath(fileName);
            String baseName = file.getFileName().toString();
            processFrames(fileName, "C:\\Users\\jstho\\GRIP\\output\\out_" + baseName);
        } catch (Exception e) {
            e.printStackTrace();
        }
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
        return processPipelineOutputs(gripPipeline);
    }

    private GripPipeline executePipeline(Mat image, double minHue, double minSaturation, double minLuminance) {
        GripPipeline gripPipeline = new GripPipeline();
        gripPipeline.setMinHue(minHue);
        gripPipeline.setMinSaturation(minSaturation);
        gripPipeline.setMinLuminance(minLuminance);

        gripPipeline.process(image);

        return gripPipeline;
    }

    private List<RotatedRect> processPipelineOutputs(GripPipeline pipeline) {
        List<MatOfPoint> contours = pipeline.findContoursOutput();
        System.out.println(String.format("Found %s contours", contours.size()));

        ArrayList<RotatedRect> filtered = new ArrayList<>();
        for(MatOfPoint contour : contours) {
            MatOfPoint2f matOfPoint2f = new MatOfPoint2f();
            contour.convertTo(matOfPoint2f, CvType.CV_32FC2);
            RotatedRect rr = Imgproc.minAreaRect(matOfPoint2f);
            double area = rr.size.area();
            double ratio = rr.size.height / rr.size.width;

            boolean inAreaRange = (area > 500) && (area < 8000);
            boolean inRatioRange = inRatioRange(ratio, 10, 8);
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
