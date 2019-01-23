package org.usfirst.frc.team3407.vision;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.videoio.VideoCapture;

import java.util.ArrayList;

public class Main {

    public static void main(String[] args) {
        new Main().run((args.length == 0)? null : args[0]);
    }

    public void run(String fileName) {
        try {
            processFrames(fileName);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void processFrames(String fileName) throws Exception {
        GripPipeline gripPipeline = new GripPipeline();

        VideoCapture videoCapture = (fileName == null) ? new VideoCapture(0) : new VideoCapture(fileName);

        System.out.println(String.format("Starting video capture for %s: open=%s", fileName, videoCapture.isOpened()));
        Mat frame = new Mat();

        int frameCount = 0;
        while (videoCapture.read(frame)) {
            System.out.println("FrameCount=" + frameCount++);
            gripPipeline.process(frame);
            processPipelineOutputs(gripPipeline.filterContoursOutput());
            Thread.sleep(5000);
        }
    }

    private void processPipelineOutputs(ArrayList<MatOfPoint> contours) {
        System.out.println(String.format("Found %s contours: %s", contours.size(), contours.toString()));

    }
}
