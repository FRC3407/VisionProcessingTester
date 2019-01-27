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

    private static final double[][] HSL = {
        {49.0, 57.0, 126.0},
        {20.0, 23.0, 90.0},
    };

    private static final Scalar BLUE = new Scalar(255, 0, 0);
    private static final Scalar GREEN = new Scalar(0, 255, 0);
    private static final Scalar RED = new Scalar(0, 0, 255);
    private static final Scalar BLACK = new Scalar(0, 0, 0);

    private static final Scalar[] COLORS = { RED, BLUE };

    private static final double MIN_AREA = 1000;
    private static final double MAX_AREA = 4000;
    private static final double TARGET_RATIO = 5.0;
    private static final double TARGET_RATIO_OFFSET = 2.5;

    private static final double RECTANGLE_PAIR_MIN_X_DIFFERENCE = 50;
    private static final double RECTANGLE_PAIR_MAX_X_DIFFERENCE = 200;
    private static final double RECTANGLE_PAIR_MAX_Y_DIFFERENCE = 10;

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

        System.out.println(String.format("\nStarting video capture for %s: open=%s", fileName, videoCapture.isOpened()));
        Mat frame = new Mat();

        int frameCount = 0;
        while (videoCapture.read(frame)) {
            System.out.println("\nFrameCount=" + frameCount++);
            System.out.println(String.format("Max-X is %s and Max-Y is %s", frame.width(), frame.height()));

            ArrayList<RotatedRect> possibeTargetMatches = new ArrayList<>();
            for (int i = 0;i < HSL.length;i++) {
                List<RotatedRect> rectangles = runPipeline(frame, HSL[i][0], HSL[i][1], HSL[i][2]);
                System.out.println(String.format("Found %s possible target matches", rectangles.size()));
                for (RotatedRect rectangle : rectangles) {
                    drawRotatedRectangle(frame, rectangle, COLORS[i]);
                }
                possibeTargetMatches.addAll(rectangles);
            }

            List<HatchTarget> hatchTargets = findHatchTargets(possibeTargetMatches);
            System.out.println(String.format("Found %s possible hatch matches", hatchTargets.size()));
            hatchTargets = filterSameHatchTargets(hatchTargets);
            System.out.println(String.format("Found %s hatch matches after removing duplicates", hatchTargets.size()));
            int targetIndex = 0;
            for (HatchTarget hatchTarget : hatchTargets) {
                drawRotatedRectangle(frame, hatchTarget.left, BLACK);
                drawRotatedRectangle(frame, hatchTarget.right, BLACK);
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

    private List<RotatedRect> runPipeline(Mat image, double minHue, double minSaturation, double minLuminance) {
        long startTime = System.currentTimeMillis();
        GripPipeline gripPipeline = executePipeline(image, minHue, minSaturation, minLuminance);
        List<RotatedRect> rotatedRects = processPipelineOutputs(gripPipeline, MIN_AREA, MAX_AREA, TARGET_RATIO,
                TARGET_RATIO_OFFSET);
        System.out.println(String.format("Pipeline ran in %s milliseconds", System.currentTimeMillis() - startTime));
        return rotatedRects;
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
                //System.out.println(String.format("Center=%s Area=%s Angle=%s Ratio=%s",
                //        rr.center, area, rr.angle, ratio));
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

    private List<HatchTarget> findHatchTargets(List<RotatedRect> targets) {
        ArrayList<HatchTarget> hatchTargets = new ArrayList<>();

        int innerEnd = targets.size();
        int outerEnd = innerEnd - 1;
        for (int i = 0; i < outerEnd; i++) {
            for (int j = i + 1; j < innerEnd; j ++) {
                boolean isHatchTarget = isHatchTarget(targets.get(i), targets.get(j));
                if (isHatchTarget) {
                    hatchTargets.add(new HatchTarget(targets.get(i), targets.get(j)));
                }
            }
        }

        return hatchTargets;
    }

    private boolean isHatchTarget(RotatedRect rr1, RotatedRect rr2) {
        Point center1 = rr1.center;
        Point center2 = rr2.center;

        double centerXDifference = Math.abs(center1.x - center2.x);
        double centerYDifference = Math.abs(center1.y - center2.y);

        //System.out.println(String.format("XDiff=%s YDiff=%s", centerXDifference, centerYDifference));

        return (centerYDifference < RECTANGLE_PAIR_MAX_Y_DIFFERENCE) &&
                (centerXDifference > RECTANGLE_PAIR_MIN_X_DIFFERENCE) &&
                (centerXDifference < RECTANGLE_PAIR_MAX_X_DIFFERENCE);
    }

    private List<HatchTarget> filterSameHatchTargets(List<HatchTarget> targets) {
        if (targets.size() < 2) {
            return targets;
        }

        ArrayList<HatchTarget> currentTargets = new ArrayList<>(targets);
        while (true) {
            ArrayList<HatchTarget> newTargets = new ArrayList<>();
            HatchTarget singleTarget = currentTargets.get(0);
            newTargets.add(singleTarget);
            for (int i = 1; i < currentTargets.size(); i++) {
                if (!isSameHatchTarget(singleTarget, currentTargets.get(i))) {
                    newTargets.add(currentTargets.get(i));
                }
            }
            if (currentTargets.size() == newTargets.size()) {
                break;
            }
            currentTargets = newTargets;
        }

        return currentTargets;
    }

    private boolean isSameHatchTarget(HatchTarget ht1, HatchTarget ht2) {
        double midpoint1 = ht1.getMidPoint();
        double midpoint2 = ht2.getMidPoint();

        return Math.abs(midpoint1 - midpoint2) < 10;
    }

    class HatchTarget {
        private RotatedRect left;
        private RotatedRect right;

        public HatchTarget(RotatedRect rr1, RotatedRect rr2) {
            if (rr1.center.x < rr2.center.x) {
                left = rr1;
                right = rr2;
            } else {
                left = rr2;
                right = rr1;
            }
        }

        public double getMidPoint() {
            return (left.center.x + right.center.x) / 2;
        }

        public double getOffset(double range) {
            double midpoint = range / 2;
            return getMidPoint() - midpoint;
        }
    }
}
