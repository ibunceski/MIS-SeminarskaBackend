//package mk.ukim.finki.dians.hw2backend.web;
//
//import mk.ukim.finki.dians.hw2backend.model.IssuerData;
//import mk.ukim.finki.dians.hw2backend.model.IssuerDates;
//import mk.ukim.finki.dians.hw2backend.service.IssuerDataService;
//import mk.ukim.finki.dians.hw2backend.service.IssuerDatesService;
//import org.springframework.http.HttpStatus;
//import org.springframework.http.ResponseEntity;
//import org.springframework.web.bind.annotation.*;
//import org.springframework.web.client.RestTemplate;
//
//import java.util.List;
//import java.util.Optional;
//
//@RestController
//@RequestMapping("/api")
//public class StockController {
//
//    final IssuerDataService issuerDataService;
//    final IssuerDatesService issuerDatesService;
//    private final RestTemplate restTemplate = new RestTemplate();
//
//    public StockController(IssuerDataService issuerDataService, IssuerDatesService issuerDatesService) {
//        this.issuerDataService = issuerDataService;
//        this.issuerDatesService = issuerDatesService;
//    }
//
//    @GetMapping("/issuers")
//    public List<String> getAllIssuers() {
//        return issuerDataService.getAllIssuers();
//    }
//
//    @GetMapping("/issuer-data")
//    public List<IssuerData> getAllIssuerData() {
//        return issuerDataService.getAllData();
//    }
//
//
//    @GetMapping("/issuer-data/{issuer}")
//    public ResponseEntity<List<IssuerData>> getDataByIssuer(@PathVariable String issuer) throws InterruptedException {
//        List<IssuerData> data = issuerDataService.getDataByIssuer(issuer);
//        return ResponseEntity.ok(data);
//    }
//
//
//    @GetMapping("/issuer-dates/{issuer}")
//    public ResponseEntity<IssuerDates> getIssuerLastDate(@PathVariable String issuer) {
//        Optional<IssuerDates> data = issuerDatesService.getLastDateForIssuer(issuer);
//        if (data.isPresent()) {
//            return ResponseEntity.ok(data.get());
//        } else {
//            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
//                    .body(null);
//        }
//    }
//
//
//    @GetMapping("/fill-data")
//    public void checkData() {
////        String url = "http://localhost:8000/api/scrape";
//        String url = "http://dians-hw-scraper:8000/api/scrape";
//        restTemplate.getForObject(url, String.class);
//    }
//}
//

package mk.ukim.finki.dians.hw2backend.web;

import mk.ukim.finki.dians.hw2backend.model.IssuerData;
import mk.ukim.finki.dians.hw2backend.model.IssuerDates;
import mk.ukim.finki.dians.hw2backend.service.IssuerDataService;
import mk.ukim.finki.dians.hw2backend.service.IssuerDatesService;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;

import java.util.List;
import java.util.Optional;

@RestController
@RequestMapping("/api")
public class StockController {

    final IssuerDataService issuerDataService;
    final IssuerDatesService issuerDatesService;
    final RestTemplate restTemplate = new RestTemplate();

    @Value("${SCRAPER_URL:http://localhost:8000}")
    private String scraper_url;

    @Value("${ANALYZER_URL:http://localhost:8001}")
    private String analyzer_url;

    public StockController(IssuerDataService issuerDataService, IssuerDatesService issuerDatesService) {
        this.issuerDataService = issuerDataService;
        this.issuerDatesService = issuerDatesService;
    }

    @GetMapping("/issuers")
    public ResponseEntity<List<String>> getAllIssuers() {
        List<String> issuers = issuerDataService.getAllIssuers();
        return ResponseEntity.ok(issuers);
    }

    @GetMapping("/issuer-data")
    public ResponseEntity<List<IssuerData>> getAllIssuerData() {
        List<IssuerData> data = issuerDataService.getAllData();
        return ResponseEntity.ok(data);
    }

    @GetMapping("/issuer-data/{issuer}")
    public ResponseEntity<List<IssuerData>> getDataByIssuer(@PathVariable String issuer) throws InterruptedException {
        List<IssuerData> data = issuerDataService.getDataByIssuer(issuer);
        if (data.isEmpty()) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(null);
        }
        return ResponseEntity.ok(data);
    }

    @GetMapping("/issuer-dates/{issuer}")
    public ResponseEntity<IssuerDates> getIssuerLastDate(@PathVariable String issuer) {
        Optional<IssuerDates> data = issuerDatesService.getLastDateForIssuer(issuer);
        return data.map(ResponseEntity::ok)
                .orElseGet(() -> ResponseEntity.status(HttpStatus.NOT_FOUND).body(null));
    }

    @GetMapping("/fill-data")
    public ResponseEntity<Void> checkData() {
        try {
            String url = scraper_url + "/api/scrape";
            restTemplate.getForObject(url, String.class);
            return ResponseEntity.ok().build();
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }

    @GetMapping("/nlp/{issuer}")
    public ResponseEntity<?> getNLP(@PathVariable String issuer) {
        String pythonNLPUrl = analyzer_url + "/api/nlp/" + issuer;
        ResponseEntity<String> response = restTemplate.getForEntity(pythonNLPUrl, String.class);
        return ResponseEntity.ok(response.getBody());
    }

    @GetMapping("/lstm/{issuer}")
    public ResponseEntity<?> getLSTM(@PathVariable String issuer) {
        String pythonLSTMUrl = analyzer_url + "/api/lstm/" + issuer;
        ResponseEntity<String> response = restTemplate.getForEntity(pythonLSTMUrl, String.class);
        return ResponseEntity.ok(response.getBody());
    }

    @GetMapping("/technical/{issuer}")
    public ResponseEntity<?> getTechnical(@PathVariable String issuer) {
        String pythonTechnicalUrl = analyzer_url + "/api/technical/" + issuer;
        ResponseEntity<String> response = restTemplate.getForEntity(pythonTechnicalUrl, String.class);
        return ResponseEntity.ok(response.getBody());
    }
}


