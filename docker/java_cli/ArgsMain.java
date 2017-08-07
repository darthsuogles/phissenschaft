import java.util.List;
import java.util.ArrayList;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.JCommander;

public class ArgsMain {
    @Parameter
    private List<String> parameters = new ArrayList<>();

    @Parameter(names = { "-log", "-verbose" }, description = "Level of verbosity")
    private Integer verbose = 1;

    @Parameter(names = "-groups", description = "Comma-separated list of group names to be run")
    private String groups;

    @Parameter(names = "-debug", description = "Debug mode")
    private boolean debug = false;

    public static void main(String ... args) {
        ArgsMain obj = new ArgsMain();
        JCommander.newBuilder()
            .addObject(obj)
            .build()
            .parse(args);
        obj.run();        
    }
    
    void run() {
        String mode = null;
        if (debug) mode = "DEBUG"; else mode = "RELEASE";
        System.out.format("%s %s %s", verbose, groups, mode);
    }
}
