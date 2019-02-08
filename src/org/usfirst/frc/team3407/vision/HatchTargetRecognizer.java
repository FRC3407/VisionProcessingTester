package org.usfirst.frc.team3407.vision;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiConsumer;

public class HatchTargetRecognizer {

    private static final double[][] HSL = {
            {49.0, 57.0, 126.0},
            {20.0, 23.0, 90.0},
    };

    private static final double MIN_AREA = 1000;
    private static final double MAX_AREA = 4000;
    private static final double TARGET_RATIO = 5.0;
    private static final double TARGET_RATIO_OFFSET = 2.5;

    private static final double RECTANGLE_PAIR_MIN_X_DIFFERENCE = 50;
    private static final double RECTANGLE_PAIR_MAX_X_DIFFERENCE = 200;
    private static final double RECTANGLE_PAIR_MAX_Y_DIFFERENCE = 10;

    public List<HatchTarget> find(Mat image, BiConsumer<RotatedRect,Integer> processConsumer) {
        ArrayList<RotatedRect> possibeTargetMatches = new ArrayList<>();
        for (int i = 0;i < HSL.length;i++) {
            List<RotatedRect> rectangles = process(image, HSL[i][0], HSL[i][1], HSL[i][2]);
            System.out.println(String.format("Found %s possible target matches", rectangles.size()));
            for (RotatedRect rectangle : rectangles) {
                processConsumer.accept(rectangle, i);
            }
            possibeTargetMatches.addAll(rectangles);
        }

        List<HatchTarget> hatchTargets = findHatchTargets(possibeTargetMatches);
        System.out.println(String.format("Found %s possible hatch matches", hatchTargets.size()));
        hatchTargets = filterSameHatchTargets(hatchTargets);
        System.out.println(String.format("Found %s hatch matches after removing duplicates", hatchTargets.size()));

        return hatchTargets;
    }

    private List<RotatedRect> process(Mat image, double minHue, double minSaturation, double minLuminance) {
        long startTime = System.currentTimeMillis();
        List<MatOfPoint> contours = executePipeline(image, minHue, minSaturation, minLuminance);
        List<RotatedRect> rotatedRects = processPipelineOutputs(contours, MIN_AREA, MAX_AREA, TARGET_RATIO,
                TARGET_RATIO_OFFSET);
        System.out.println(String.format("Pipeline ran in %s milliseconds", System.currentTimeMillis() - startTime));
        return rotatedRects;
    }

    private List<MatOfPoint> executePipeline(Mat image, double minHue, double minSaturation, double minLuminance) {
        GripPipeline gripPipeline = new GripPipeline();
        gripPipeline.setMinHue(minHue);
        gripPipeline.setMinSaturation(minSaturation);
        gripPipeline.setMinLuminance(minLuminance);

        gripPipeline.process(image);

        return gripPipeline.findContoursOutput();
    }

    private List<RotatedRect> processPipelineOutputs(List<MatOfPoint> contours, double minArea, double maxArea,
                                                     double targetRatio, double targetRatioOffset) {
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

        public RotatedRect getLeft() {
            return left;
        }

        public RotatedRect getRight() {
            return right;
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
