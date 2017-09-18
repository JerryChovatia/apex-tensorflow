package com.github.ambarishpande.ApexTensorflow;

/**
 * Created by ambarish on 28/8/17.
 */

import java.io.IOException;
import java.io.InputStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.apache.commons.io.IOUtils;
import org.apache.hadoop.fs.Path;
import com.datatorrent.api.Context;
import com.datatorrent.api.DefaultOutputPort;
import com.datatorrent.lib.io.fs.AbstractFileInputOperator;

/**
 * This in imageInBytes file reader operator specifically for image processing.
 * It emits imageInBytes Data POJO which contains the image data as byte array and the file name as String.
 * Data POJO is uniform across all operators under image.
 * Image processing can be very CPU intensive and hence time consuming. It is recommended to use the slowDown=true
 * property and appropriately set slowDownMills (per tuple) to avoid back pressure on down stream operators.
 */

@org.apache.hadoop.classification.InterfaceStability.Evolving
public class ImageReader extends AbstractFileInputOperator<Data>
{
  public static final char START_FILE = '(';
  public static final char FINISH_FILE = ')';
  private static final Logger LOG = LoggerFactory.getLogger(ImageReader.class);
  public final transient DefaultOutputPort<Data> output = new DefaultOutputPort<Data>();
  public final transient DefaultOutputPort<byte[]> outputBytes = new DefaultOutputPort<>();
  public int countImageSent = 0;
  public int countImageSent2 = 0;
  public transient String filePathStr;
  private byte[] imageInBytes;
  private boolean stop;
  private transient int pauseTime;
  private transient Path filePath;
  private Boolean slowDown = true;
  private long slowDownMills = 100;
  public transient String fileType = "";
  private String[] compatibleFileTypes = {"jpg", "png", "jpeg", "fits", "gif", "tif"};

  public Boolean getSlowDown()
  {
    return slowDown;
  }

  /**
   * Set the boolean value from properties.xml. True will slow down emit tuple by set milliseconds.
   * Default value is true.
   * @param slowDown
   */
  public void setSlowDown(Boolean slowDown)
  {
    this.slowDown = slowDown;
  }

  public long getSlowDownMills()
  {
    return slowDownMills;
  }

  /**
   * Set the interval between two emit tuple calls in milliseconds.
   * Default value is 100 ms.
   * @param slowDownMills
   */
  public void setSlowDownMills(long slowDownMills)
  {
    this.slowDownMills = slowDownMills;
  }

  public boolean setFileType()
  {
    for (int i = 0; i < compatibleFileTypes.length; i++) {
      if ( filePath.toString().toLowerCase().endsWith(compatibleFileTypes[i])) {
        fileType = compatibleFileTypes[i];
        if ( fileType.equalsIgnoreCase("jpeg")) {
          fileType = "jpg";
        }
      }
    }
    if ( fileType.isEmpty() ) {
      return false;
    } else {
      return true;
    }
  }

  @Override
  public void setup(Context.OperatorContext context)
  {
    super.setup(context);
    pauseTime = context.getValue(Context.OperatorContext.SPIN_MILLIS);

    if (null != filePathStr) {      // restarting from checkpoint
      filePath = new Path(filePathStr);
    }
  }

  @Override
  public void emitTuples()
  {
    if (!stop) {        // normal processing
      super.emitTuples();
      return;
    }
    // we have end-of-file, so emit no further tuples till next window; relax for imageInBytes bit
    try {
      Thread.sleep(pauseTime);
    } catch (InterruptedException e) {
      throw new RuntimeException("Sleep interrupted" + e);
    }
  }

  @Override
  public void endWindow()
  {
    super.endWindow();
    stop = false;
  }

  @Override
  protected InputStream openFile(Path curPath) throws IOException
  {
    LOG.debug("openFile: curPath = {}", curPath);
    filePath = curPath;
    filePathStr = filePath.toString();
    LOG.info("readOpen " + START_FILE + filePath.getName());
    setFileType();
    InputStream is = super.openFile(filePath);
    if (!filePathStr.contains(".fits")) {
      imageInBytes = IOUtils.toByteArray(is);
    } else {
      String fitsPath = filePath.getParent().toString() + "/" + filePath.getName();
      if (fitsPath.contains(":")) {
        fitsPath = fitsPath.replace("file:", "");
      }
    }
    return is;
  }

  @Override
  protected void closeFile(InputStream is) throws IOException
  {
    LOG.debug("closeFile: filePath = {}", filePath);
    super.closeFile(is);
    LOG.info("readClose " + filePath.getName() + FINISH_FILE);
    filePath = null;
    stop = true;
  }

  @Override
  protected Data readEntity() throws IOException
  {
    if (countImageSent < 1) {
      countImageSent++;
      Data data = new Data();
      data.bytesImage = imageInBytes;
      data.fileName = filePath.getName().toString();
      data.imageType = fileType;
      return data;
    }
    LOG.debug("readEntity: EOF for {}", filePath);
    countImageSent = 0;
    return null;
  }

  @Override
  protected void emit(Data data)
  {
    if (slowDown) {
      Boolean loop = true;
      long startTime = System.currentTimeMillis();
      while (loop == true) {
        long currentTime = System.currentTimeMillis();
        if (currentTime - startTime >= slowDownMills) {
          loop = false;
        }
      }
    }
    output.emit(data);
//    outputBytes.emit(data.bytesImage);
    countImageSent2++;
  }

}