package com.mogoweb.nsfw;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Environment;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
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
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import com.mogoweb.nsfw.env.Logger;
import com.mogoweb.nsfw.env.Utility;
import com.mogoweb.nsfw.tflite.Classifier;
import com.mogoweb.nsfw.tflite.Classifier.Device;
import com.mogoweb.nsfw.tflite.Classifier.Model;

import java.io.File;
import java.io.FileFilter;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.Arrays;
import java.util.List;

public class MainActivity extends AppCompatActivity
        implements ActivityCompat.OnRequestPermissionsResultCallback,
                    AdapterView.OnItemSelectedListener {
    private static final Logger LOGGER = new Logger();

    private static final int REQUEST_CAMERA = 0;
    private static final int SELECT_FILE = 1;

    private static final String TAG = "AIDog";

    private Classifier classifier;

    private TextView tvResult;
    private Spinner modelSpinner;
    private Spinner deviceSpinner;

    private View mLayout;

    private Handler handler;
    private HandlerThread handlerThread;

    private Model model = Model.QUANTIZED;
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
                if (Utility.checkCameraPermission(MainActivity.this)) {
                    // Permission is already available, start camera
                    cameraIntent();
                }
            }
        });
        Button btnSelectPhoto = (Button)findViewById(R.id.btnSelectPhoto);
        btnSelectPhoto.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                boolean result = Utility.checkStoragePermission(MainActivity.this);
                if (result)
                    galleryIntent();
            }
        });
        tvResult = (TextView)findViewById(R.id.tvResult);

        Button btnBenchmarkImages = (Button)findViewById(R.id.btnBenchmarkImages);
        btnBenchmarkImages.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                boolean result = Utility.checkStoragePermission(MainActivity.this);
                if (result) {
                    benchmarkIntent();
                }
            }
        });

        modelSpinner = findViewById(R.id.model_spinner);
        modelSpinner.setOnItemSelectedListener(this);
        deviceSpinner= findViewById(R.id.device_spinner);
        deviceSpinner.setOnItemSelectedListener(this);

        recreateClassifier(getModel(), getDevice(), getNumThreads());
    }

    @Override
    public synchronized void onResume() {
        LOGGER.d("onResume " + this);
        super.onResume();

        handlerThread = new HandlerThread("inference");
        handlerThread.start();
        handler = new Handler(handlerThread.getLooper());
    }

    @Override
    public synchronized void onPause() {
        LOGGER.d("onPause " + this);

        handlerThread.quitSafely();
        try {
            handlerThread.join();
            handlerThread = null;
            handler = null;
        } catch (final InterruptedException e) {
            LOGGER.e(e, "Exception!");
        }

        super.onPause();
    }

    @Override
    public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {
        if (adapterView == modelSpinner) {
            setModel(Model.valueOf(adapterView.getItemAtPosition(i).toString().toUpperCase()));
        } else if (adapterView == deviceSpinner) {
            setDevice(Device.valueOf(adapterView.getItemAtPosition(i).toString()));
        }
    }

    @Override
    public void onNothingSelected(AdapterView<?> adapterView) {

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

    private void benchmarkIntent()
    {
        File extStore = Environment.getExternalStorageDirectory();
        // ==> /storage/emulated/0/note.txt
        String imagesPath = extStore.getAbsolutePath() + "/images";
        File directory = new File(imagesPath);
        if (!directory.exists()) {
            showToast("/sdcard/images directory not exists");
            return;
        }
        File[] files = directory.listFiles(new FileFilter() {
            @Override
            public boolean accept(File file) {
                return (file.getPath().endsWith(".jpg") || file.getPath().endsWith(".jpeg"));
            }
        });
        LOGGER.i("images: "+ files.length);
        if (files.length == 0) {
            showToast("no jpeg images in /sdcard/images");
            return;
        }

        Arrays.sort(files);
        runInBackground(
            new Thread(new Runnable() {
                public void run() {
                    try {
                        File extStore = Environment.getExternalStorageDirectory();
                        File outFile = new File(extStore.getAbsolutePath(), "results.txt");
                        FileWriter out = new FileWriter(outFile);
                        out.append("File\tSFW Score\tNSFW Score\n");

                        long startTime = SystemClock.uptimeMillis();
                        for (File imageFile : files) {
                            Bitmap bitmap = BitmapFactory.decodeFile(imageFile.getAbsolutePath());
                            final List<Classifier.Recognition> results = classifier.recognizeImage(bitmap);
                            if (results != null) {
                                out.append(String.format("%s\t%f\t%f\n", imageFile.getName(),
                                        results.get(0).getConfidence(), results.get(1).getConfidence()));
                            }
                        }
                        long endTime = SystemClock.uptimeMillis();
                        LOGGER.i("Timecost to benchmark: " + (endTime - startTime));
                        out.flush();
                        out.close();
                    } catch (FileNotFoundException e) {
                        e.printStackTrace();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }));
    }

    protected synchronized void runInBackground(final Runnable r) {
        if (handler != null) {
            handler.post(r);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        if (requestCode == Utility.MY_PERMISSIONS_REQUEST_CAMERA) {
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
        } else if (requestCode == Utility.MY_PERMISSIONS_REQUEST_READ_EXTERNAL_STORAGE) {
            if (grantResults.length == 1 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // Permission has been granted. Start camera.
                benchmarkIntent();
            } else {
                // Permission request was denied.
                Snackbar.make(mLayout, R.string.storage_permission_denied,
                        Snackbar.LENGTH_SHORT)
                        .show();
            }
        }
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
        Toast.makeText(this, s, Toast.LENGTH_SHORT).show();
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
            runInBackground(() -> recreateClassifier(getModel(), getDevice(), getNumThreads()));
        }
    }

    protected Device getDevice() {
        return device;
    }

    private void setDevice(Device device) {
        if (this.device != device) {
            LOGGER.d("Updating  device: " + device);
            this.device = device;
            runInBackground(() -> recreateClassifier(getModel(), getDevice(), getNumThreads()));
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
