package com.github.ambarishpande.ApexTensorflow;

/**
 * Created by ambarish on 21/9/17.
 */
public class TfCategory
{
  private int categoryIndex;
  private String categoryName;
  private float probabilty;

  public TfCategory(int categoryIndex, String categoryName, float probabilty)
  {
    this.categoryIndex = categoryIndex;
    this.categoryName = categoryName;
    this.probabilty = probabilty;
  }

  public TfCategory()
  {
  }

  public int getCategoryIndex()
  {
    return categoryIndex;
  }

  public void setCategoryIndex(int categoryIndex)
  {
    this.categoryIndex = categoryIndex;
  }

  public String getCategoryName()
  {
    return categoryName;
  }

  public void setCategoryName(String categoryName)
  {
    this.categoryName = categoryName;
  }

  public double getProbabilty()
  {
    return probabilty;
  }

  public void setProbabilty(float probabilty)
  {
    this.probabilty = probabilty;
  }

  @Override
  public String toString()
  {
    return "TfCategory{" +
      "categoryIndex=" + categoryIndex +
      ", categoryName='" + categoryName + '\'' +
      ", probabilty=" + probabilty +
      '}';
  }
}
