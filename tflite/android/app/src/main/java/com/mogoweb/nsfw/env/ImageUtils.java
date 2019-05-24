package com.mogoweb.nsfw.env;

import android.graphics.Bitmap;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class ImageUtils {
    public static Bitmap scaleBitmap(Bitmap bitmap, int width, int height) {
        Mat mat = new Mat();
        //add alpha
        Utils.bitmapToMat(bitmap.copy(Bitmap.Config.ARGB_8888, false), mat, true);

        Mat mat1 = new Mat();
        Imgproc.resize(mat, mat1, new Size(width, height), 0, 0, Imgproc.INTER_LINEAR);
        //add alpha
        Bitmap scaledBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        //convert
        Utils.matToBitmap(mat1, scaledBitmap);

        return scaledBitmap;
    }
}
