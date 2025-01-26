package mk.ukim.finki.dians.hw2backend.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDate;

@Entity
@Table(name = "issuer_data")
@AllArgsConstructor
@Data
@NoArgsConstructor
@IdClass(IssuerDataKey.class)
public class IssuerData {

    @Id
    @JsonProperty("issuer")
    private String issuer;

    @Id
    @JsonProperty("date")
    private LocalDate date;

    @JsonProperty("lastTradePrice")
    private String lastTradePrice;

    @JsonProperty("maxPrice")
    private String maxPrice;

    @JsonProperty("minPrice")
    private String minPrice;

    @JsonProperty("avgPrice")
    private String avgPrice;

    @JsonProperty("percentChange")
    private String percentChange;

    @JsonProperty("volume")
    private String volume;

    @JsonProperty("turnoverBest")
    private String turnoverBest;

    @JsonProperty("totalTurnover")
    private String totalTurnover;

    @Override
    public String toString() {
        return "IssuerData{" +
                "issuer='" + issuer + '\'' +
                ", date=" + date +
                ", lastTradePrice='" + lastTradePrice + '\'' +
                ", maxPrice='" + maxPrice + '\'' +
                ", minPrice='" + minPrice + '\'' +
                ", avgPrice='" + avgPrice + '\'' +
                ", percentChange='" + percentChange + '\'' +
                ", volume='" + volume + '\'' +
                ", turnoverBest='" + turnoverBest + '\'' +
                ", totalTurnover='" + totalTurnover + '\'' +
                '}';
    }
}