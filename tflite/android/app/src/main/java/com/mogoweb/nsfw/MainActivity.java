package com.mogoweb.nsfw;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.annotation.UiThread;
import android.support.design.widget.Snackbar;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.text.SpannableString;
import android.text.SpannableStringBuilder;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import com.mogoweb.nsfw.env.Logger;
import com.mogoweb.nsfw.env.Utility;
import com.mogoweb.nsfw.tflite.Classifier;
import com.mogoweb.nsfw.tflite.Classifier.Device;
import com.mogoweb.nsfw.tflite.Classifier.Model;

import java.io.IOException;
import java.util.List;

public class MainActivity extends AppCompatActivity
        implements ActivityCompat.OnRequestPermissionsResultCallback {
    private static final Logger LOGGER = new Logger();

    private static final int REQUEST_CAMERA = 0;
    private static final int SELECT_FILE = 1;

    private static final int PERMISSION_REQUEST_CAMERA = 0;

    private static final String TAG = "AIDog";

    private Classifier classifier;

    private TextView tvResult;

    private View mLayout;

    private Model model = Model.FLOAT;
    private Device device = Device.CPU;
    private int numThreads = -1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mLayout = findViewById(R.id.container);

        Button btnTakePhoto = (Button)findViewById(R.id.btnTakePhoto);
        btnTakePhoto.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Check if the Camera permission has been granted
                if (ActivityCompat.checkSelfPermission(MainActivity.this, Manifest.permission.CAMERA)
                        == PackageManager.PERMISSION_GRANTED) {
                    // Permission is already available, start camera
                    cameraIntent();
                } else {
                    // Permission is missing and must be requested.
                    requestCameraPermission();
                }
            }
        });
        Button btnSelectPhoto = (Button)findViewById(R.id.btnSelectPhoto);
        btnSelectPhoto.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                boolean result = Utility.checkPermission(MainActivity.this);
                if (result)
                    galleryIntent();
            }
        });
        tvResult = (TextView)findViewById(R.id.tvResult);

        recreateClassifier(getModel(), getDevice(), getNumThreads());
    }

    /**
     * Requests the {@link android.Manifest.permission#CAMERA} permission.
     * If an additional rationale should be displayed, the user has to launch the request from
     * a SnackBar that includes additional information.
     */
    private void requestCameraPermission() {
        // Permission has not been granted and must be requested.
        if (ActivityCompat.shouldShowRequestPermissionRationale(this,
                Manifest.permission.CAMERA)) {
            // Provide an additional rationale to the user if the permission was not granted
            // and the user would benefit from additional context for the use of the permission.
            // Display a SnackBar with cda button to request the missing permission.
            Snackbar.make(mLayout, R.string.camera_access_required,
                    Snackbar.LENGTH_INDEFINITE).setAction(R.string.ok, new View.OnClickListener() {
                @Override
                public void onClick(View view) {
                    // Request the permission
                    ActivityCompat.requestPermissions(MainActivity.this,
                            new String[]{Manifest.permission.CAMERA},
                            PERMISSION_REQUEST_CAMERA);
                }
            }).show();

        } else {
            // Request the permission. The result will be received in onRequestPermissionResult().
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA}, PERMISSION_REQUEST_CAMERA);
        }
    }

    private void cameraIntent()
    {
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(intent, REQUEST_CAMERA);
    }

    private void galleryIntent()
    {
        Intent intent = new Intent();
        intent.setType("image/*");
        intent.setAction(Intent.ACTION_GET_CONTENT); //
        startActivityForResult(Intent.createChooser(intent, "Select File"), SELECT_FILE);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        // BEGIN_INCLUDE(onRequestPermissionsResult)
        if (requestCode == PERMISSION_REQUEST_CAMERA) {
            // Request for camera permission.
            if (grantResults.length == 1 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // Permission has been granted. Start camera.
                cameraIntent();
            } else {
                // Permission request was denied.
                Snackbar.make(mLayout, R.string.camera_permission_denied,
                        Snackbar.LENGTH_SHORT)
                        .show();
            }
        }
        // END_INCLUDE(onRequestPermissionsResult)
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == Activity.RESULT_OK) {
            if (requestCode == SELECT_FILE)
                onSelectFromGalleryResult(data);
            else if (requestCode == REQUEST_CAMERA)
                onCaptureImageResult(data);
        }
    }

    /**
     * Shows a {@link Toast} on the UI thread for the classification results.
     *
     * @param s The message to show
     */
    private void showToast(String s) {
        SpannableStringBuilder builder = new SpannableStringBuilder();
        SpannableString str1 = new SpannableString(s);
        builder.append(str1);
        showToast(builder);
    }

    private void showToast(final SpannableStringBuilder builder) {
        runOnUiThread(
                new Runnable() {
                    @Override
                    public void run() {
                        tvResult.setText(builder, TextView.BufferType.SPANNABLE);
                    }
                });
    }

    @SuppressWarnings("deprecation")
    private void onSelectFromGalleryResult(Intent data) {
        if (data != null) {
            try {
                Bitmap bitmap = MediaStore.Images.Media.getBitmap(getApplicationContext().getContentResolver(), data.getData());
                Bitmap bm = Bitmap.createScaledBitmap(bitmap, classifier.getImageSizeX(), classifier.getImageSizeX(), false);
                final List<Classifier.Recognition> results = classifier.recognizeImage(bm);
                showResults(results);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private void onCaptureImageResult(Intent data) {
        Bitmap bm = (Bitmap) data.getExtras().get("data");

        final List<Classifier.Recognition> results = classifier.recognizeImage(bm);
        showResults(results);
    }

    @UiThread
    protected void showResults(List<Classifier.Recognition> results) {
        if (results != null) {
            String res = "";
            Classifier.Recognition recognition = results.get(0);
            if (recognition != null) {
                if (recognition.getTitle() != null)
                    res = res + recognition.getTitle() + ":";
                if (recognition.getConfidence() != null)
                    res = res + String.format("%.2f", (100 * recognition.getConfidence())) + "% ";
            }

            Classifier.Recognition recognition1 = results.get(1);
            if (recognition1 != null) {
                if (recognition1.getTitle() != null)
                    res = res + recognition1.getTitle() + ":";
                if (recognition1.getConfidence() != null)
                    res = res + String.format("%.2f", (100 * recognition1.getConfidence())) + "%";
            }
            tvResult.setText(res);
        }
    }

    private void recreateClassifier(Model model, Device device, int numThreads) {
        if (classifier != null) {
            LOGGER.d("Closing classifier.");
            classifier.close();
            classifier = null;
        }
        if (device == Device.GPU && model == Model.QUANTIZED) {
            LOGGER.d("Not creating classifier: GPU doesn't support quantized models.");
            runOnUiThread(
                    () -> {
                        Toast.makeText(this, "GPU does not yet supported quantized models.", Toast.LENGTH_LONG)
                                .show();
                    });
            return;
        }
        try {
            LOGGER.d(
                    "Creating classifier (model=%s, device=%s, numThreads=%d)", model, device, numThreads);
            classifier = Classifier.create(this, model, device, numThreads);
        } catch (IOException e) {
            LOGGER.e(e, "Failed to create classifier.");
        }
    }

    protected Model getModel() {
        return model;
    }

    private void setModel(Model model) {
        if (this.model != model) {
            LOGGER.d("Updating  model: " + model);
            this.model = model;
        }
    }

    protected Device getDevice() {
        return device;
    }

    private void setDevice(Device device) {
        if (this.device != device) {
            LOGGER.d("Updating  device: " + device);
            this.device = device;
            final boolean threadsEnabled = device == Device.CPU;
        }
    }

    protected int getNumThreads() {
        return numThreads;
    }

    private void setNumThreads(int numThreads) {
        if (this.numThreads != numThreads) {
            LOGGER.d("Updating  numThreads: " + numThreads);
            this.numThreads = numThreads;
        }
    }
}
