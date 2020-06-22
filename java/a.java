abstract class Phone{
  public abstract void showConfig();
}
class Iphone extends Phone{
  public void showConfig(){
    System.out.println("Iphone");
  }
}
class Android extends Phone{
  public void showConfig(){
    System.out.println("Android");
  }
}
class a{
  public static void main(String[] args){
    Phone android=new Android();
    Phone iphone=new Iphone();

    android.showConfig();
    iphone.showConfig();
  }
}
