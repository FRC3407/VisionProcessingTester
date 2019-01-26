package org.usfirst.frc.team3407.vision;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import java.util.ArrayList;
import java.util.List;

public class Main {

    private static final Scalar RED = new Scalar(255, 0, 0);

    public static void main(String[] args) {
        new Main().run((args.length == 0)? null : args[0]);
    }

    public void run(String fileName) {
        try {
            processFrames(fileName, "C:\\Users\\jstho\\GRIP\\output\\out.jpg");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void processFrames(String fileName, String outputFileName) throws Exception {
        GripPipeline gripPipeline = new GripPipeline();

        VideoCapture videoCapture = (fileName == null) ? new VideoCapture(0) : new VideoCapture(fileName);

        System.out.println(String.format("Starting video capture for %s: open=%s", fileName, videoCapture.isOpened()));
        Mat frame = new Mat();

        int frameCount = 0;
        while (videoCapture.read(frame)) {
            System.out.println("FrameCount=" + frameCount++);
            gripPipeline.process(frame);
            List<Point[]> contours = processPipelineOutputs(gripPipeline);
            System.out.println(String.format("Found %s possible target matches", contours.size()));
            for(Point[] outlinePoints : contours) {
                for (int i = 0;i < 4;i++) {
                    Imgproc.line(frame, outlinePoints[i], outlinePoints[(i + 1)%4], RED, 3);
                }
            }
            Imgcodecs.imwrite(outputFileName, frame);
            Thread.sleep(5000);
        }
    }

    private List<Point[]> processPipelineOutputs(GripPipeline pipeline) {
        List<MatOfPoint> contours = pipeline.findContoursOutput();
        System.out.println(String.format("Found %s contours", contours.size()));

        ArrayList<Point[]> filtered = new ArrayList<>();
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
                Point[] points = new Point[4];
                rr.points(points);
                filtered.add(points);
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
