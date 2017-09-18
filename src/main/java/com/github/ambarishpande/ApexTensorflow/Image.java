package com.github.ambarishpande.ApexTensorflow;

import org.tensorflow.Tensor;

/**
 * Created by ambarish on 18/9/17.
 */
public class Image
{
  private Tensor image;
  private Data d;

  public Image(Tensor image, Data d)
  {
    this.image = image;
    this.d = d;
  }

  public Image()
  {
  }

  public Tensor getImage()
  {
    return image;
  }

  public void setImage(Tensor image)
  {
    this.image = image;
  }

  public Data getD()
  {
    return d;
  }

  public void setD(Data d)
  {
    this.d = d;
  }
}
