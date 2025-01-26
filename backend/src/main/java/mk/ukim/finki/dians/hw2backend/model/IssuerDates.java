package mk.ukim.finki.dians.hw2backend.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDate;

@Entity
@Data
@Table(name = "issuer_dates")
@AllArgsConstructor
@NoArgsConstructor
public class IssuerDates {

    @Id
    @JsonProperty("issuer")
    private String issuer;

    @JsonProperty("lastDate")
    private LocalDate lastDate;
}
