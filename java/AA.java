interface A{
public abstract void a();
public abstract void b();
public abstract void c();
public abstract void d();
}
class AA implements A{
public static void main(String args[]){
  A obj=new AA();
  obj.a();
  obj.b();

}
@Override
public void a(){
System.out.println("I am a");
}
@Override
public void b(){
System.out.println("I am b");
}
@Override
public void c(){
System.out.println("I am c");
}
@Override
public void d(){
System.out.println("I am d");
}
}
